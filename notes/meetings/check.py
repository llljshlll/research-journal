"""
Shading Bake 파이프라인:
  카메라 이미지(Gemini 보정본)를 UV 텍스처 공간에 직접 투영.
  조명 시뮬레이션·gradient descent 없음.
  결과 텍스처에 촬영 당시 조명·그림자가 그대로 베이크됨.

  pipeline_albedo.py 와의 차이:
    - run_refine (inverse rendering) → bake_view (직접 픽셀 복사)
    - shading map 나눗셈 없음 (조명 제거 안 함)
    - 여러 뷰의 기여를 가중 평균으로 누적
    - 미커버 텍셀은 최종 단계에서 nearest-neighbor 채움

결과: output/pipeline/{RUN_ID}/
"""

import json
import shutil
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  상수
# ─────────────────────────────────────────────────────────────
RUN_ID           = "26_shading_bake_5view"

IMAGE_SIZE       = 512
MESH_PATH        = "unwrap/mesh_unwrap.obj"
JSON_PATH        = "data/sequence_5.json"

FIXED_LIGHT_VIEW_NAME = "view_00001"
INIT_TARGET_IMAGE     = "gemini_denoising/output/data_013FHA7K_ring3_00001.png"
INIT_REF_VIEW_NAME    = "view_00001"

COVERAGE_DILATION        = 1
COVERAGE_ALPHA_THRESHOLD = 0.5

GEMINI_CACHE_DIR = "output/pipeline/11_aov_visibility_mask_bounce8"

# ─────────────────────────────────────────────────────────────
#  출력 경로
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(f"output/pipeline/{RUN_ID}")
BASE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  공통 유틸
# ─────────────────────────────────────────────────────────────
def load_view_params(json_path, view_name):
    with open(json_path, "r") as f:
        data = json.load(f)
    for v in data["views"]:
        if v["view_name"] == view_name:
            T_cw = np.array(v["extrinsic"]["T_cw"], dtype=np.float64)
            K    = np.array(v["intrinsic"]["K"],    dtype=np.float64)
            R       = T_cw[:3, :3]
            cam_pos = T_cw[:3, 3]
            forward = -R[:, 2]
            up      = R[:, 1]
            target  = cam_pos + forward
            fx  = K[0, 0]
            cx  = K[0, 2]
            cy  = K[1, 2]
            fov_x = float(np.degrees(2.0 * np.arctan(IMAGE_SIZE / (2.0 * fx))))
            ppx   = float((cx - IMAGE_SIZE / 2.0) / IMAGE_SIZE)
            ppy   = float(-(cy - IMAGE_SIZE / 2.0) / IMAGE_SIZE)
            return {
                "origin":    cam_pos.tolist(),
                "target":    target.tolist(),
                "up":        up.tolist(),
                "fov":       fov_x,
                "ppx":       ppx,
                "ppy":       ppy,
                "view_name": view_name,
            }
    raise ValueError(f"View '{view_name}' not found in {json_path}")


def load_view_order(json_path: str) -> list:
    with open(json_path, "r") as f:
        data = json.load(f)
    return [v["view_name"] for v in data["views"]]


# ─────────────────────────────────────────────────────────────
#  Gemini 캐시 조회
# ─────────────────────────────────────────────────────────────
def find_cached_gemini(view_name: str) -> Path | None:
    cache_base = Path(GEMINI_CACHE_DIR)
    if not cache_base.exists():
        return None
    for d in sorted(cache_base.iterdir()):
        if d.is_dir() and d.name.endswith(f"_{view_name}"):
            img = d / f"gemini_{view_name}.png"
            if img.exists():
                return img
    return None


# ─────────────────────────────────────────────────────────────
#  핵심: 한 뷰를 UV 텍스처에 베이크
# ─────────────────────────────────────────────────────────────
def bake_view(
    target_image_path: str,
    view_name: str,
    light_dir: list,
    accum_np: np.ndarray,   # (H, W, 3) 누적 색상합 (in-place 수정)
    weight_np: np.ndarray,  # (H, W)    누적 픽셀 수 (in-place 수정)
    spp: int = 256,
):
    """
    카메라 뷰 이미지 → UV 텍스처 공간에 직접 복사.

    AOV(uv:uv) 렌더로 각 카메라 픽셀의 UV 좌표를 얻고,
    target_image의 해당 픽셀 색상을 텍스처에 누적한다.
    조명 계산 없음. 색상은 target_image 그대로 (shading baked).
    """
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")

    view = load_view_params(JSON_PATH, view_name)
    H = W = IMAGE_SIZE

    # ── AOV 렌더: 카메라 픽셀 → UV 좌표 매핑 ──────────────────
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {
            "type": "aov",
            "aovs": "uv:uv",
            "integrator": {"type": "direct"},
        },
        "sun": {
            "type": "directional",
            "direction": light_dir,
            "irradiance": {"type": "spectrum", "value": 5.0},
        },
        "world": {
            "type": "constant",
            "radiance": {"type": "spectrum", "value": 0.5},
        },
        "sensor": {
            "type": "perspective",
            "fov_axis": "x",
            "fov":  view["fov"],
            "principal_point_offset_x": view["ppx"],
            "principal_point_offset_y": view["ppy"],
            "to_world": mi.ScalarTransform4f.look_at(
                origin=view["origin"], target=view["target"], up=view["up"]
            ),
            "film": {
                "type": "hdrfilm",
                "width": W, "height": H,
                "pixel_format": "rgba",
            },
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "tooth": {
            "type": "obj",
            "filename": MESH_PATH,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "spectrum", "value": 1.0},
            },
        },
    })

    img_np = np.array(mi.render(scene, spp=spp))  # (H, W, 6): RGB, alpha, U, V

    alpha  = img_np[:, :, 3]
    u_vals = img_np[:, :, 4]
    v_vals = img_np[:, :, 5]

    # ── 타겟 이미지 로드 (색상 소스) ──────────────────────────
    bmp_target = mi.Bitmap(target_image_path).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False
    )
    target_np = np.array(bmp_target)  # (H, W, 3), [0, 1]

    # ── 가시 카메라 픽셀 선택 ─────────────────────────────────
    visible = alpha > COVERAGE_ALPHA_THRESHOLD   # (H, W) bool

    cam_rows, cam_cols = np.where(visible)

    # 카메라 픽셀 → UV 텍셀 좌표
    tex_col = np.clip((u_vals[cam_rows, cam_cols] * W).astype(int), 0, W - 1)
    tex_row = np.clip((v_vals[cam_rows, cam_cols] * H).astype(int), 0, H - 1)

    # 타겟 이미지 색상 추출
    colors = target_np[cam_rows, cam_cols]  # (N, 3)

    # ── UV 텍스처에 누적 (같은 텍셀에 여러 뷰가 기여하면 평균) ──
    np.add.at(accum_np,  (tex_row, tex_col), colors)
    np.add.at(weight_np, (tex_row, tex_col), 1.0)

    n_px    = int(visible.sum())
    n_texel = len(np.unique(np.stack([tex_row, tex_col], axis=1), axis=0))
    print(f"  [BAKE] view={view_name}: 카메라 픽셀={n_px}  커버 텍셀={n_texel}/{H*W} "
          f"({100*n_texel/(H*W):.1f}%)")


# ─────────────────────────────────────────────────────────────
#  최종 텍스처 완성 (평균화 + 미커버 텍셀 채움)
# ─────────────────────────────────────────────────────────────
def finalize_texture(accum_np: np.ndarray, weight_np: np.ndarray) -> np.ndarray:
    """
    누적합 / 누적 수 → 평균 색상.
    한 번도 커버되지 않은 텍셀은 nearest-neighbor 로 채움.
    """
    H, W = accum_np.shape[:2]
    filled = weight_np > 0                          # (H, W) bool

    baked = np.where(
        filled[:, :, np.newaxis],
        accum_np / np.maximum(weight_np[:, :, np.newaxis], 1.0),
        0.0,
    ).astype(np.float32)

    # 미커버 텍셀 → nearest-neighbor 채움
    n_unfilled = int((~filled).sum())
    if n_unfilled > 0:
        from scipy.ndimage import distance_transform_edt
        _, nearest = distance_transform_edt(~filled, return_indices=True)
        rows_nn = nearest[0][~filled]
        cols_nn = nearest[1][~filled]
        baked[~filled] = baked[rows_nn, cols_nn]
        print(f"  [FINALIZE] 미커버 텍셀 {n_unfilled}/{H*W} → nearest-neighbor 채움")
    else:
        print(f"  [FINALIZE] 전체 텍셀 커버 완료")

    return baked


# ─────────────────────────────────────────────────────────────
#  중간 결과 PNG 저장 (디버그용)
# ─────────────────────────────────────────────────────────────
def save_baked_png(baked_np: np.ndarray, path: Path):
    import mitsuba as mi
    # shading baked 텍스처는 이미 [0,1] sRGB 이므로 tonemap 없이 저장
    rgb   = np.clip(baked_np, 0.0, 1.0).astype(np.float32)
    alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
    bmp = mi.Bitmap(np.concatenate([rgb, alpha], axis=2))
    bmp = bmp.convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, srgb_gamma=False)
    bmp.write(str(path))


# ─────────────────────────────────────────────────────────────
#  메인 파이프라인
# ─────────────────────────────────────────────────────────────
def compute_fixed_light_dir(json_path, view_name):
    p = load_view_params(json_path, view_name)
    origin = np.array(p["origin"])
    target = np.array(p["target"])
    d = target - origin
    d = d / np.linalg.norm(d)
    print(f"[FIXED LIGHT] view={view_name}  dir={d.tolist()}")
    return d.tolist()


def main():
    print(f"\n{'#'*60}")
    print(f"#  Shading Bake 파이프라인  RUN_ID={RUN_ID}")
    print(f"#  조명 없음 — 카메라 이미지 → UV 직접 투영")
    print(f"{'#'*60}")

    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")

    fixed_light_dir = compute_fixed_light_dir(JSON_PATH, FIXED_LIGHT_VIEW_NAME)

    all_views = load_view_order(JSON_PATH)
    print(f"[뷰 순서] 총 {len(all_views)}개: {all_views}")

    try:
        init_idx = all_views.index(INIT_REF_VIEW_NAME)
    except ValueError:
        raise RuntimeError(f"INIT_REF_VIEW_NAME='{INIT_REF_VIEW_NAME}' 이 JSON 에 없습니다.")

    H = W = IMAGE_SIZE
    accum_np  = np.zeros((H, W, 3), dtype=np.float64)  # 누적 색상합
    weight_np = np.zeros((H, W),    dtype=np.float64)  # 누적 픽셀 수

    # ── 초기 뷰 베이크 ────────────────────────────────────────
    print(f"\n[BAKE INIT] view={INIT_REF_VIEW_NAME}  source={INIT_TARGET_IMAGE}")
    bake_view(
        target_image_path = INIT_TARGET_IMAGE,
        view_name         = INIT_REF_VIEW_NAME,
        light_dir         = fixed_light_dir,
        accum_np          = accum_np,
        weight_np         = weight_np,
    )

    # ── 나머지 뷰 순서대로 베이크 ────────────────────────────
    remaining_views = all_views[init_idx + 1:] + all_views[:init_idx]

    for loop_idx, view_name in enumerate(remaining_views, start=1):
        print(f"\n{'*'*50}")
        print(f"*  LOOP {loop_idx}/{len(remaining_views)}   뷰={view_name}")
        print(f"{'*'*50}")

        # Gemini 캐시 조회 → 없으면 이 뷰 스킵
        cached = find_cached_gemini(view_name)
        if cached is None:
            print(f"  [SKIP] Gemini 캐시 없음 — 실사 이미지 없이 베이크 불가")
            continue

        print(f"  [SOURCE] {cached}")
        bake_view(
            target_image_path = str(cached),
            view_name         = view_name,
            light_dir         = fixed_light_dir,
            accum_np          = accum_np,
            weight_np         = weight_np,
        )

        # 중간 결과 저장
        interim = finalize_texture(accum_np.copy(), weight_np.copy())
        save_baked_png(interim, BASE_DIR / f"baked_after_loop{loop_idx:04d}_{view_name}.png")

    # ── 최종 텍스처 저장 ──────────────────────────────────────
    print(f"\n[FINALIZE] 최종 텍스처 생성 중...")
    baked_np = finalize_texture(accum_np, weight_np)

    # EXR (선형, 원본값 그대로)
    exr_path = BASE_DIR / "baked_texture.exr"
    mi.Bitmap(baked_np).write(str(exr_path))

    # PNG (시각화용)
    png_path = BASE_DIR / "baked_texture.png"
    save_baked_png(baked_np, png_path)

    # 커버리지 맵 저장
    coverage_vis = np.clip(weight_np / max(weight_np.max(), 1), 0, 1).astype(np.float32)
    coverage_3c  = np.stack([coverage_vis] * 3, axis=2)
    mi.Bitmap(coverage_3c).write(str(BASE_DIR / "coverage_map.exr"))

    n_covered = int((weight_np > 0).sum())
    print(f"\n{'#'*60}")
    print(f"#  베이크 완료")
    print(f"#  커버리지: {n_covered}/{H*W} texels ({100*n_covered/(H*W):.1f}%)")
    print(f"#  EXR: {exr_path}")
    print(f"#  PNG: {png_path}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
