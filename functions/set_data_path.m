function [data_dir, out_dir, log_dir, aux_dir] = set_data_path()
% Locate the folders used by demand estimation.
%   data_dir: local folder with df.parquet (data/final_estimation_data/demand)
%   out_dir, log_dir: local output folders (data/outputs/demand/OutData and Log)
%   aux_dir: Dropbox folder with the auxiliary input tables (household size,
%            income distribution, micro moments, national deflator)

% <repo>/codes/demand_estimation/functions/set_data_path.m -> <repo>
repo_dir = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));

data_dir = fullfile(repo_dir, 'data', 'final_estimation_data', 'demand');
out_dir = fullfile(repo_dir, 'data', 'outputs', 'demand', 'OutData');
log_dir = fullfile(repo_dir, 'data', 'outputs', 'demand', 'Log');

if ~exist(data_dir, 'dir')
    error(['Local data folder not found: ', data_dir]);
end

for d = {out_dir, log_dir}
    if ~exist(d{1}, 'dir')
        mkdir(d{1});
    end
end

% Auxiliary input tables in Dropbox (set DROPBOX_ROOT if Dropbox is not in a standard location)
aux_subdir = fullfile('GS_Network', 'Data', 'final_estimation_data', 'demand');
dropbox_candidates = {
    getenv('DROPBOX_ROOT')
    fullfile(getenv('HOME'), 'Dropbox')
    fullfile(getenv('HOME'), 'Library', 'CloudStorage', 'Dropbox')
    fullfile(getenv('USERPROFILE'), 'Dropbox')
    fullfile(getenv('USERPROFILE'), 'Dropbox (Personal)')
    fullfile(getenv('USERPROFILE'), 'Dropbox (Business)')
};

aux_dir = '';
for i = 1:length(dropbox_candidates)
    candidate = fullfile(dropbox_candidates{i}, aux_subdir);
    if ~isempty(dropbox_candidates{i}) && exist(candidate, 'dir')
        aux_dir = candidate;
        break
    end
end

if isempty(aux_dir)
    error(['Could not find ', aux_subdir, ' inside Dropbox. You can set DROPBOX_ROOT manually.']);
end

end
