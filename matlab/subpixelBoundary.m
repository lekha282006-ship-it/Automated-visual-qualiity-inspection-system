function contours = subpixelBoundary(binaryImg, scale)
%SUBPIXELBOUNDARY Find boundaries with sub-pixel coordinates by upscaling
%   contours = subpixelBoundary(binaryImg, scale)
%   binaryImg: logical or uint8 mask
%   scale: integer upscale factor (default 4)
%   contours: cell array of Nx2 arrays of [x,y] coordinates (double)

if nargin < 2
    scale = 4;
end

if ~islogical(binaryImg)
    bw = im2bw(binaryImg);
else
    bw = binaryImg;
end

% Upscale
bwBig = imresize(bw, scale, 'nearest');

% Find boundaries
B = bwboundaries(bwBig, 'noholes');
contours = cell(size(B));
for k = 1:numel(B)
    pts = B{k}; % [row, col]
    % convert to x,y and scale back
    x = pts(:,2) / scale;
    y = pts(:,1) / scale;
    contours{k} = [x, y];
end
end
