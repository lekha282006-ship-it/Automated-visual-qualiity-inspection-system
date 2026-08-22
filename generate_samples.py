import cv2
import numpy as np
import os

def draw_washer(img, center=(250, 250), outer_r=150, inner_r=50, color=180, bg_color=30):
    cv2.circle(img, center, outer_r, color, -1)
    cv2.circle(img, center, inner_r, bg_color, -1)
    return img

def create_synthetic_parts():
    os.makedirs('sample_data/test_images', exist_ok=True)
    
    width, height = 500, 500
    bg_color = 30
    part_color = 180
    
    # 0. Golden Reference
    ref_img = np.full((height, width), bg_color, dtype=np.uint8)
    ref_img = draw_washer(ref_img, outer_r=150, inner_r=50, color=part_color, bg_color=bg_color)
    cv2.imwrite('sample_data/reference.png', ref_img)
    
    # 1. Good Part
    good_img = ref_img.copy()
    # add slight gaussian noise for realism without triggering defects
    noise = np.random.normal(0, 1, good_img.shape)
    good_img = np.clip(good_img.astype(float) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite('sample_data/test_images/1_good_part.png', good_img)
    
    # 2. Undersized Contour
    undersized = np.full((height, width), bg_color, dtype=np.uint8)
    undersized = draw_washer(undersized, outer_r=138, inner_r=50, color=part_color, bg_color=bg_color)
    cv2.imwrite('sample_data/test_images/2_undersized.png', undersized)
    
    # 3. Oversized Contour
    oversized = np.full((height, width), bg_color, dtype=np.uint8)
    oversized = draw_washer(oversized, outer_r=162, inner_r=50, color=part_color, bg_color=bg_color)
    cv2.imwrite('sample_data/test_images/3_oversized.png', oversized)
    
    # 4. Surface Scratch
    scratch = ref_img.copy()
    cv2.line(scratch, (150, 200), (220, 280), 80, 3) # dark scratch
    cv2.line(scratch, (280, 150), (320, 190), 220, 2) # light scratch
    cv2.imwrite('sample_data/test_images/4_scratch.png', scratch)
    
    # 5. Surface Chip/Crack
    chip = ref_img.copy()
    cv2.circle(chip, (110, 180), 25, bg_color, -1)
    cv2.imwrite('sample_data/test_images/5_chip.png', chip)
    
    # 6. Hole Misalignment
    misaligned = np.full((height, width), bg_color, dtype=np.uint8)
    misaligned = draw_washer(misaligned, center=(250, 250), outer_r=150, inner_r=50, color=part_color, bg_color=bg_color)
    cv2.circle(misaligned, (250, 250), 50, part_color, -1) # fill original hole
    cv2.circle(misaligned, (270, 230), 50, bg_color, -1) # new off-center hole
    cv2.imwrite('sample_data/test_images/6_misaligned_hole.png', misaligned)
    
    # 7. Burrs/Flash Edge Flaws
    burr = ref_img.copy()
    cv2.circle(burr, (390, 250), 15, part_color, -1)
    cv2.circle(burr, (250, 95), 12, part_color, -1)
    cv2.imwrite('sample_data/test_images/7_burrs.png', burr)
    
    print("Synthetic SPC samples generated in sample_data/test_images/")

if __name__ == "__main__":
    create_synthetic_parts()
