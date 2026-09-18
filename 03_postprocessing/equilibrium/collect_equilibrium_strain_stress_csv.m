%% 提取并复制所有工况的 CSV 文件
clear; clc;

%% ===================== 用户设置 =====================
rootDir = "C:\Users\Admin01\Desktop\stage3_tensile_refined_multiH\phase1_y_tension_results";
csvName = "strain_stress.csv";

% 设置目标文件夹（保存所有复制出的CSV文件）
targetDir = fullfile(rootDir, "_all_csv_files");
if ~exist(targetDir, "dir"); mkdir(targetDir); end

%% ===================== 扫描工况目录 =====================
d = dir(rootDir);
caseDirs = d([d.isdir]);
% 排除当前目录、父目录和输出目录
caseDirs = caseDirs(~ismember({caseDirs.name}, {'.', '..', '_all_csv_files'}));

count = 0; % 记录成功复制的文件数量

for i = 1:numel(caseDirs)
    caseName = string(caseDirs(i).name);
    
    % 检查文件夹名称是否符合工况格式
    if ~isValidCaseName(caseName); continue; end

    srcCsv = fullfile(rootDir, caseName, csvName);
    if ~isfile(srcCsv); continue; end

    % 为防止同名覆盖，目标文件命名为 "工况名_原文件名.csv"
    destCsv = fullfile(targetDir, caseName + "_" + csvName);
    
    % 执行复制
    ok = safeCopyFile(srcCsv, destCsv, 5, 0.5);
    if ok
        count = count + 1;
        fprintf("已复制: %s\n", destCsv);
    else
        fprintf("复制失败: %s\n", srcCsv);
    end
end

fprintf("\n操作完成！共成功复制 %d 个文件至文件夹: %s\n", count, targetDir);

%% ===================== 局部函数 =====================
function tf = isValidCaseName(caseName)
    % 修改正则匹配：涵盖 0H, 1H 以及 005cov 到 100cov 的所有形式
    tf = ~isempty(regexp(string(caseName), '^sigma(3|5|11|17)_GB_(0H|1H|\d{3}cov)$', 'once'));
end

function ok = safeCopyFile(src, dst, maxTry, pauseSec)
    ok = false;
    for k = 1:maxTry
        try
            copyfile(src, dst, "f");
            info = dir(dst); 
            if ~isempty(info) && info.bytes > 0
                ok = true; 
                return; 
            end
        catch
            pause(pauseSec); 
        end
    end
end