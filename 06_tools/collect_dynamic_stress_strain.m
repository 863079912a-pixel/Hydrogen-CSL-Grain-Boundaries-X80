%% collect_dynamic_stress_strain.m
% 批量提取 4 类晶界 × 9 种氢覆盖度的动态应力-应变 CSV 文件
%
% 原始目录结构示例：
% D:\桌面\所有动态的轨迹文件\
%   3.3.3动态0,1\sigma3\0H\results\stress_strain_dynamic.csv
%   3.3.3动态5,10,15,20\sigma3\020cov\results\stress_strain_dynamic.csv
%   3.3.3动态25,50,100\sigma3\100cov\results\stress_strain_dynamic.csv
%
% 输出：
% D:\桌面\所有动态的轨迹文件\Dynamic_Stress_Strain_Collected\
%   Sigma3_0H_stress_strain_dynamic.csv
%   Sigma3_1H_stress_strain_dynamic.csv
%   Sigma3_5H_stress_strain_dynamic.csv
%   ...
%   Sigma17_100H_stress_strain_dynamic.csv
%
% 同时生成：
%   extraction_manifest.csv
%
% 说明：
% - 只复制并重命名，不修改 CSV 内部数据。
% - 理论上应提取 4 × 9 = 36 个文件。
% - 若有缺失文件，会在命令窗口明确提示。

clear; clc;

%% 1. 根目录与输出目录
rootDir = 'D:\桌面\所有动态的轨迹文件';

outDir = fullfile(rootDir, 'Dynamic_Stress_Strain_Collected');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% 2. 晶界类型
sigmaFolders = {'sigma3', 'sigma5', 'sigma11', 'sigma17'};
sigmaNames   = {'Sigma3', 'Sigma5', 'Sigma11', 'Sigma17'};

%% 3. 氢覆盖度与对应父目录
% sourceCov: 原始文件夹名
% outCov   : 输出文件名中的覆盖度名称
% groupDir : 对应的动态计算父目录

sourceCov = {'0H', '1H', ...
             '005cov', '010cov', '015cov', '020cov', ...
             '025cov', '050cov', '100cov'};

outCov = {'0H', '1H', ...
          '5H', '10H', '15H', '20H', ...
          '25H', '50H', '100H'};

groupDir = {'3.3.3动态0,1', '3.3.3动态0,1', ...
            '3.3.3动态5,10,15,20', '3.3.3动态5,10,15,20', ...
            '3.3.3动态5,10,15,20', '3.3.3动态5,10,15,20', ...
            '3.3.3动态25,50,100', '3.3.3动态25,50,100', ...
            '3.3.3动态25,50,100'};

%% 4. 批量提取
expectedCount = numel(sigmaFolders) * numel(sourceCov);
foundCount = 0;
missingCount = 0;

manifest = cell(expectedCount, 6);
row = 0;

fprintf('============================================================\n');
fprintf('开始提取动态应力-应变文件\n');
fprintf('根目录：%s\n', rootDir);
fprintf('输出目录：%s\n', outDir);
fprintf('理论文件数：%d\n', expectedCount);
fprintf('============================================================\n\n');

for i = 1:numel(sigmaFolders)
    for j = 1:numel(sourceCov)

        row = row + 1;

        srcFile = fullfile( ...
            rootDir, ...
            groupDir{j}, ...
            sigmaFolders{i}, ...
            sourceCov{j}, ...
            'results', ...
            'stress_strain_dynamic.csv');

        newName = sprintf('%s_%s_stress_strain_dynamic.csv', ...
            sigmaNames{i}, outCov{j});

        dstFile = fullfile(outDir, newName);

        if exist(srcFile, 'file')
            try
                copyfile(srcFile, dstFile, 'f');
                status = 'FOUND_AND_COPIED';
                foundCount = foundCount + 1;
                fprintf('[OK]   %-10s %-6s -> %s\n', ...
                    sigmaNames{i}, outCov{j}, newName);
            catch ME
                status = ['COPY_FAILED: ' ME.message];
                missingCount = missingCount + 1;
                fprintf('[FAIL] %-10s %-6s 复制失败\n', ...
                    sigmaNames{i}, outCov{j});
            end
        else
            status = 'MISSING';
            missingCount = missingCount + 1;
            fprintf('[MISS] %-10s %-6s\n', sigmaNames{i}, outCov{j});
            fprintf('       %s\n', srcFile);
        end

        manifest{row,1} = sigmaNames{i};
        manifest{row,2} = outCov{j};
        manifest{row,3} = sourceCov{j};
        manifest{row,4} = srcFile;
        manifest{row,5} = dstFile;
        manifest{row,6} = status;
    end
end

%% 5. 输出检查清单
T = cell2table(manifest, 'VariableNames', { ...
    'Sigma', ...
    'Coverage', ...
    'SourceFolder', ...
    'SourceFile', ...
    'OutputFile', ...
    'Status'});

manifestFile = fullfile(outDir, 'extraction_manifest.csv');
writetable(T, manifestFile, 'Encoding', 'UTF-8');

%% 6. 最终统计
fprintf('\n============================================================\n');
fprintf('提取完成\n');
fprintf('成功：%d / %d\n', foundCount, expectedCount);
fprintf('缺失或失败：%d\n', missingCount);
fprintf('检查清单：%s\n', manifestFile);
fprintf('============================================================\n');

if foundCount == expectedCount
    fprintf('\n全部 36 个动态应力-应变文件均已成功提取并重命名。\n');
else
    fprintf('\n注意：并非全部 36 个文件都被找到，请查看 extraction_manifest.csv。\n');
end
