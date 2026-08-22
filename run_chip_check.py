import cv2, numpy as np
from src.inspector import PartInspector

def draw_washer(img, center=(250,250), outer_r=120, inner_r=40, color=180, bg_color=30):
    cv2.circle(img, center, outer_r, color, -1)
    cv2.circle(img, center, inner_r, bg_color, -1)
    return img

w,h=400,400
bg=30
ref=np.full((h,w),bg,dtype=np.uint8)
ref=draw_washer(ref, center=(250,250))

test=ref.copy()
cv2.circle(test, (200,200), 15, bg, -1)

insp=PartInspector(reference=ref, size_tol=5.0, defect_thresh=10)
metrics, ann = insp.inspect(test)
print(metrics)
print('defects:', metrics.get('defects'))
