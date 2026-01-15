# MV-Adapter + Skyfall-GS with Mesh Prior

## Motivation
- MV-Adapter produces only 6 views
- Direct 3DGS optimization is unstable with sparse views
- Skyfall-GS alleviates sparse-view issues
- We further exploit the availability of GT mesh

## Overall Strategy
1. Use MV-Adapter to generate 6 canonical views
2. Initialize 3DGS using mesh-based point sampling
3. Replace Skyfall-GS geometry ambiguity with strong mesh prior
4. Restrict densification and optimization on mesh surface

## Key Design Choices
- Initialization: mesh-based Gaussian placement
- Rotation: mesh normal-aligned
- Depth: GT depth or pseudo-depth
- Densification: mesh-aware split / prune / clone

See details in:
- `mesh_prior.md`
