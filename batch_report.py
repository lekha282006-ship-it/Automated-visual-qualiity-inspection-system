import os
import csv
import cv2
from src.inspector import PartInspector
from src.db import InspectionDB

def run_batch():
    os.makedirs('results', exist_ok=True)
    
    inspector = PartInspector('sample_data/reference.png')
    db = InspectionDB()
    
    test_dir = 'sample_data/test_images'
    results = []
    
    for filename in os.listdir(test_dir):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        test_path = os.path.join(test_dir, filename)
        result = inspector.inspect(test_path)
        
        annotated_path = os.path.join('results', f"annotated_{filename}")
        if result['annotated_image'] is not None:
            cv2.imwrite(annotated_path, result['annotated_image'])
            
        db.log_result(filename, result)
        
        results.append({
            'Image': filename,
            'Status': result['status'],
            'Reason': result['reason'],
            'Annotated File': annotated_path
        })
        
        icon = "PASS" if result['status'] == "PASS" else "FAIL"
        print(f"{filename}\t{icon}\t{result['reason']}")
        
    with open('results/inspection_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Image', 'Status', 'Reason', 'Annotated File'])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    run_batch()
