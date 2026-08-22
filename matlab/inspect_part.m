function result = inspect_part(refImgPath, testImgPath)
% INSPECT_PART Placeholder MATLAB inspection script.
% Loads reference and test images, performs a simple alignment and difference,
% and returns a struct with fields similar to the Python inspector.

ref = imread(refImgPath);
test = imread(testImgPath);

% convert to grayscale
if size(ref,3) == 3
    refGray = rgb2gray(ref);
else
    refGray = ref;
end
if size(test,3) == 3
    testGray = rgb2gray(test);
else
    testGray = test;
end

% Simple registration using phase correlation (placeholder)
try
    tform = imregcorr(testGray, refGray);
    aligned = imwarp(testGray, tform, 'OutputView', imref2d(size(refGray)));
catch
    aligned = testGray;
end

% difference and threshold
D = imabsdiff(refGray, aligned);
th = imbinarize(D, graythresh(D));

% compute simple metrics
totalDefectArea = sum(th(:));
numObjects = bwconncomp(th).NumObjects;

result = struct();
result.status = 'PASS';
if totalDefectArea > 1000
    result.status = 'FAIL';
end
result.total_defect_area = double(totalDefectArea);
result.defect_count = numObjects;

end
