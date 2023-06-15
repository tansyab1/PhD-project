% function calculates the NIQE, BRISQUE, and Decrete Entropy of the
% video sequence and returns the mean of the each metric.

% Inputs:
%   video: video sequence to be evaluated
%   metric: metric to be calculated (niqe, brisque, or entropy)

% Outputs:
%   metric_mean: mean of the metric calculated for the video sequence

function [niqe_mean, brisque_mean, entropy_mean] = calnoref(video)

    % read video
    v = VideoReader(video);
    % get number of frames
    numFrames = v.NumberOfFrames;

    % initialize arrays
    niqe_list = zeros(1, numFrames);
    brisque_list = zeros(1, numFrames);
    entropy_list = zeros(1, numFrames);

    % loop through each frame
    for i = 1:numFrames
        % read frame
        frame = read(v, i);
        
        % if frame is not null
        if ~isempty(frame)
            % calculate niqe
            niqe_list(i) = niqe(frame);

            % calculate brisque
            brisque_list(i) = brisque(frame);

            % calculate entropy
            entropy_list(i) = entropy(frame);

    end

    % calculate mean of each metric
    niqe_mean = mean(niqe_list);
    brisque_mean = mean(brisque_list);
    entropy_mean = mean(entropy_list);
    
end