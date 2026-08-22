function [aligned, H] = alignImages(ref, test)
%ALIGNIMAGES Align test image to reference using feature matching and projective transform
%   [aligned, H] = alignImages(ref, test)
%   ref, test: 2D grayscale images (uint8 or double)
%   aligned: uint8 aligned image (same size as ref)
%   H: 3x3 projective transform matrix mapping points in test -> ref

% Ensure images are grayscale doubles internally
if ~ismatrix(ref) || ~ismatrix(test)
    error('reshape inputs to 2D grayscale images');
end
refI = im2single(ref);
testI = im2single(test);

% Choose feature detector depending on availability
useORB = exist('detectORBFeatures','file') == 2;
useBRISK = exist('detectBRISKFeatures','file') == 2;

if useORB
    pts1 = detectORBFeatures(refI);
    pts2 = detectORBFeatures(testI);
elseif useBRISK
    pts1 = detectBRISKFeatures(refI);
    pts2 = detectBRISKFeatures(testI);
else
    % fallback to SURF if available
    pts1 = detectSURFFeatures(refI);
    pts2 = detectSURFFeatures(testI);
end

% Extract features
[feat1, validPts1] = extractFeatures(refI, pts1);
[feat2, validPts2] = extractFeatures(testI, pts2);

% Match features
indexPairs = matchFeatures(feat1, feat2, 'Unique', true);

if size(indexPairs, 1) < 4
    % not enough matches: return identity
    H = eye(3);
    aligned = im2uint8(testI);
    return
end

matchedPts1 = validPts1(indexPairs(:,1));
matchedPts2 = validPts2(indexPairs(:,2));

% Estimate projective transform robustly
try
    tform = estimateGeometricTransform2D(matchedPts2, matchedPts1, 'projective', 'MaxNumTrials', 2000, 'Confidence', 99.9, 'MaxDistance', 4);
catch
    % fallback to affine
    tform = estimateGeometricTransform2D(matchedPts2, matchedPts1, 'affine', 'MaxNumTrials', 2000);
end

% Convert transform to homography matrix
Hmat = tform.T; % 3x3
H = double(Hmat);

% Warp test image to ref size
outputView = imref2d(size(ref));
warped = imwarp(testI, tform, 'OutputView', outputView);
aligned = im2uint8(warped);
end
