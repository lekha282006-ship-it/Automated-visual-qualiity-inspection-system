import cv2, numpy as np
from src.inspector import PartInspector

def draw_washer(img, center=(200,200), outer_r=150, inner_r=50, color=180, bg_color=30):
    cv2.circle(img, center, outer_r, color, -1)
    cv2.circle(img, center, inner_r, bg_color, -1)
    return img

w,h=400,400
bg=30
ref=np.full((h,w),bg,dtype=np.uint8)
ref=draw_washer(ref, outer_r=120, inner_r=40, color=180, bg_color=bg)

test=ref.copy()
cv2.circle(test, (110,180), 15, bg, -1)

insp=PartInspector(reference=ref, size_tol=5.0, defect_thresh=10)
metrics, ann = insp.inspect(test)
print('metrics:', metrics)
print('defects:', metrics.get('defects'))
for d in metrics.get('defects',[]):
    print(d)
# extra debug: inspect internal segmentation
ref_proc = insp.reference_image
test_proc = insp._preprocess(test)
th, contours = insp._segment_defects(ref_proc, test_proc)
print('segmented pixels:', int(np.count_nonzero(th)))
print('contours found:', len(contours))
for i,c in enumerate(contours):
    print(i, 'area_px=', cv2.contourArea(c))
# raw diff check
diff_raw = cv2.absdiff(insp.reference_raw, test)
print('raw diff unique vals:', np.unique(diff_raw)[:10])
print('raw diff max/min:', int(diff_raw.max()), int(diff_raw.min()))
diff_pre = cv2.absdiff(ref_proc, test_proc)
print('pre diff max/min:', int(diff_pre.max()), int(diff_pre.min()))
print('reference_raw equals test?', np.array_equal(insp.reference_raw, test))
print('ref pixel at chip:', insp.reference_raw[180,110], 'test pixel at chip:', test[180,110])
