function results = spc_analysis(values, target, tol)
% SPC_ANALYSIS Calculate capability indices and apply WECO rules
% values: numeric vector
% target: nominal target value
% tol: half-tolerance (USL = target+tol, LSL = target-tol)

if nargin < 3
    error('spc_analysis requires values, target, tol');
end

vals = double(values(:));
N = numel(vals);
if N == 0
    results = struct();
    return;
end

mu = mean(vals);
if N > 1
    sigma_sample = std(vals, 1); % MATLAB's std default is population (normalize by N-1 with flag)
    sigma_sample = std(vals, 0); % use sample std (N-1) by default? we'll use built-in unbiased
    sigma_sample = std(vals, 1); % to be explicit: use sample std (N-1) is std(vals,0)
    sigma_sample = std(vals,0);
    sigma_pop = std(vals,1);
else
    sigma_sample = 0;
    sigma_pop = 0;
end

USL = target + tol;
LSL = target - tol;

if sigma_sample > 0
    Cp = (USL - LSL) / (6 * sigma_sample);
    Cpk = min((USL - mu) / (3 * sigma_sample), (mu - LSL) / (3 * sigma_sample));
else
    Cp = 0; Cpk = 0;
end
if sigma_pop > 0
    Pp = (USL - LSL) / (6 * sigma_pop);
    Ppk = min((USL - mu) / (3 * sigma_pop), (mu - LSL) / (3 * sigma_pop));
else
    Pp = 0; Ppk = 0;
end

% Control limits
UCL = mu + 3 * sigma_sample;
LCL = mu - 3 * sigma_sample;

% WECO rules: 1) one point beyond 3σ
signals = {};
for i=1:N
    if abs(vals(i)-mu) > 3*sigma_sample
        signals{end+1} = sprintf('Rule1: point %d beyond 3σ (value=%g)', i, vals(i));
    end
end

% Rule 2: two of three beyond 2σ on same side
for i=1:(N-2)
    w = vals(i:i+2)-mu;
    if sum(w>2*sigma_sample) >= 2
        signals{end+1} = sprintf('Rule2: window %d-%d two of three >+2σ', i, i+2);
    end
    if sum(w<-2*sigma_sample) >= 2
        signals{end+1} = sprintf('Rule2: window %d-%d two of three <-2σ', i, i+2);
    end
end

% Rule 3: four of five beyond 1σ on same side
for i=1:(N-4)
    w = vals(i:i+4)-mu;
    if sum(w>1*sigma_sample) >= 4
        signals{end+1} = sprintf('Rule3: window %d-%d four of five >+1σ', i, i+4);
    end
    if sum(w<-1*sigma_sample) >= 4
        signals{end+1} = sprintf('Rule3: window %d-%d four of five <-1σ', i, i+4);
    end
end

% Rule 4: nine consecutive on same side of mean
for i=1:(N-8)
    w = vals(i:i+8);
    if all(w>mu)
        signals{end+1} = sprintf('Rule4: 9 points %d-%d above mean', i, i+8);
    end
    if all(w<mu)
        signals{end+1} = sprintf('Rule4: 9 points %d-%d below mean', i, i+8);
    end
end

results = struct();
results.mean = mu;
results.std = sigma_sample;
results.Cp = Cp;
results.Cpk = Cpk;
results.Pp = Pp;
results.Ppk = Ppk;
results.UCL = UCL;
results.LCL = LCL;
results.weco_signals = signals;

end
