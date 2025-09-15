# Arichitecture

Project Overview

This project proposes a new algorithmic framework for creating unified and editable indoor–outdoor 3D models directly from photographs. The framework integrates scale fusion and segmentation into the reconstruction process, producing a consistent model that captures both interior and exterior elements.
	•	Segment Anything Model (SAM): Provides robust image recognition and segmentation, enabling extraction of key “portal features” such as doors and windows that connect indoor and outdoor spaces.
	•	VGGT: Converts 2D images into 3D point clouds, forming the geometric foundation of the model.
	•	Point-Voxel CNN (PVCNN): With its hybrid point–voxel architecture, PVCNN captures both local set-level details and global spatial context, enabling precise mesh construction of architectural components within the fused model.

Together, these modules form an end-to-end modeling pipeline that unifies indoor and outdoor environments into a seamless, editable 3D representation.

<div style="display: flex; justify-content: center; gap: 20px; text-align: center;">

  <div>
    <img src="https://github.com/user-attachments/assets/a8162ad7-3f91-4354-bf10-014c869dc8da" width="400"/>
    <div style="font-size: 12px; color: gray; text-align: center;">
      Result of SAM semantic segmentation on the building
    </div>
  </div>

  <div>
    <img src="https://github.com/user-attachments/assets/48c11b72-ee5f-4dbe-820e-962a94346902" width="400"/>
    <div style="font-size: 12px; color: gray; text-align: center;">
      Semantic segmentation results converted into semantic 3D point clouds using VGGT
    </div>
  </div>

</div>
