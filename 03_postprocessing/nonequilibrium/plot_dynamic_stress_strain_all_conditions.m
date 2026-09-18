%% Dynamic stress-strain plotting for all manuscript conditions
% Reads the compact repository CSV files directly from:
%   05_stress_strain_data/nonequilibrium/
%
% Required columns:
%   strain_nominal
%   stress
%
% No additional stress correction, zero shifting, or smoothing is applied.
% Two figure groups are generated:
%   A. Same hydrogen condition, different grain boundaries
%   B. Same grain boundary, different hydrogen conditions

clear; clc; close all;

%% ===================== Repository paths =====================
scriptDir = fileparts(mfilename("fullpath"));
repoRoot = fileparts(fileparts(scriptDir));
dataDir = fullfile(repoRoot, "05_stress_strain_data", "nonequilibrium");
outDir = fullfile(scriptDir, "figures");

if ~isfolder(dataDir)
    error("Data directory not found: %s", dataDir);
end
if ~exist(outDir, "dir")
    mkdir(outDir);
end

summaryCsv = fullfile(outDir, "dynamic_read_status_summary.csv");

%% ===================== Case settings =====================
sigmas = ["Sigma3", "Sigma5", "Sigma11", "Sigma17"];
sigmaLabels = ["\Sigma3", "\Sigma5", "\Sigma11", "\Sigma17"];

conditions = ["0H", "1H", "5H", "10H", "15H", "20H", "25H", "50H", "100H"];
conditionLabels = ["0H", "1H", "5%", "10%", "15%", "20%", "25%", "50%", "100%"];

strainColRequired = "strain_nominal";
stressColRequired = "stress";

%% ===================== Plot settings =====================
fontName = "Times New Roman";
fontSizeAxis = 10;
fontSizeLegend = 9;
fontSizeCaption = 10;
lineWidthMain = 1.6;
axesLineWidth = 0.8;

xLabelText = "Nominal strain";
yLabelText = "Tensile stress (GPa)";

xlimManual = [];
ylimManual = [];

figA_png = fullfile(outDir, "same_hydrogen_condition_different_grain_boundaries.png");
figA_pdf = fullfile(outDir, "same_hydrogen_condition_different_grain_boundaries.pdf");
figB_png = fullfile(outDir, "same_grain_boundary_different_hydrogen_conditions.png");
figB_pdf = fullfile(outDir, "same_grain_boundary_different_hydrogen_conditions.pdf");

%% ===================== Read data =====================
data = struct();
statusRows = table();

for i = 1:numel(sigmas)
    for j = 1:numel(conditions)
        sigma = sigmas(i);
        cond = conditions(j);
        caseKey = makeCaseKey(sigma, cond);
        fileName = sigma + "_" + cond + "_stress_strain_dynamic.csv";
        csvFile = fullfile(dataDir, fileName);

        row = table();
        row.caseName = sigma + "_" + cond;
        row.csv_file = string(csvFile);
        row.read_status = "NOT_STARTED";
        row.strain_column = strainColRequired;
        row.stress_column = stressColRequired;
        row.n_points = 0;
        row.final_strain = NaN;
        row.max_stress_GPa = NaN;
        row.note = "";

        if ~isfile(csvFile)
            row.read_status = "FILE_MISSING";
            row.note = "CSV file not found.";
            statusRows = [statusRows; row]; %#ok<AGROW>
            continue;
        end

        try
            T = readtable(csvFile, "VariableNamingRule", "preserve");
        catch ME
            row.read_status = "READ_FAILED";
            row.note = "readtable failed: " + string(ME.message);
            statusRows = [statusRows; row]; %#ok<AGROW>
            continue;
        end

        names = string(T.Properties.VariableNames);
        if ~any(names == strainColRequired)
            row.read_status = "FAILED_NO_STRAIN_COLUMN";
            row.note = "Required column not found: " + strainColRequired;
            statusRows = [statusRows; row]; %#ok<AGROW>
            continue;
        end
        if ~any(names == stressColRequired)
            row.read_status = "FAILED_NO_STRESS_COLUMN";
            row.note = "Required column not found: " + stressColRequired;
            statusRows = [statusRows; row]; %#ok<AGROW>
            continue;
        end

        strain = double(T.(strainColRequired));
        stress = double(T.(stressColRequired));
        valid = isfinite(strain) & isfinite(stress);
        strain = strain(valid);
        stress = stress(valid);

        if numel(strain) < 2
            row.read_status = "FAILED_TOO_FEW_POINTS";
            row.note = "Too few valid points after filtering.";
            statusRows = [statusRows; row]; %#ok<AGROW>
            continue;
        end

        [strain, idx] = sort(strain);
        stress = stress(idx);

        data.(caseKey).strain = strain;
        data.(caseKey).stress = stress;

        row.read_status = "OK";
        row.n_points = numel(strain);
        row.final_strain = strain(end);
        row.max_stress_GPa = max(stress);
        row.note = "Repository columns used directly; no additional smoothing or stress correction.";
        statusRows = [statusRows; row]; %#ok<AGROW>
    end
end

writetable(statusRows, summaryCsv);

%% ===================== Figure A =====================
figA = figure("Color", "w", "Position", [50 50 1200 1050]);
tiledlayout(3, 3, "TileSpacing", "loose", "Padding", "compact");

for j = 1:numel(conditions)
    ax = nexttile;
    hold(ax, "on");

    for i = 1:numel(sigmas)
        caseKey = makeCaseKey(sigmas(i), conditions(j));
        if isfield(data, caseKey)
            plot(ax, data.(caseKey).strain, data.(caseKey).stress, ...
                "LineWidth", lineWidthMain, "DisplayName", sigmaLabels(i));
        end
    end

    formatAxes(ax, xLabelText, yLabelText, fontName, fontSizeAxis, axesLineWidth, xlimManual, ylimManual);
    legend(ax, "Location", "best", "Box", "on", "FontName", fontName, ...
        "FontSize", fontSizeLegend, "Interpreter", "tex");
    putSubcaption(ax, sprintf("(%s) %s hydrogen condition", char('a' + j - 1), conditionLabels(j)), ...
        fontName, fontSizeCaption);
    hold(ax, "off");
end

putFigureCaption(figA, ...
    "Dynamic stress-strain responses of different grain boundaries under the same hydrogen condition", ...
    fontName, 13);
exportgraphics(figA, figA_png, "Resolution", 400);
exportgraphics(figA, figA_pdf, "ContentType", "vector");

%% ===================== Figure B =====================
figB = figure("Color", "w", "Position", [80 80 1050 850]);
tiledlayout(2, 2, "TileSpacing", "loose", "Padding", "compact");

for i = 1:numel(sigmas)
    ax = nexttile;
    hold(ax, "on");

    for j = 1:numel(conditions)
        caseKey = makeCaseKey(sigmas(i), conditions(j));
        if isfield(data, caseKey)
            plot(ax, data.(caseKey).strain, data.(caseKey).stress, ...
                "LineWidth", lineWidthMain, "DisplayName", conditionLabels(j));
        end
    end

    formatAxes(ax, xLabelText, yLabelText, fontName, fontSizeAxis, axesLineWidth, xlimManual, ylimManual);
    legend(ax, "Location", "best", "Box", "on", "FontName", fontName, ...
        "FontSize", fontSizeLegend, "Interpreter", "tex");
    putSubcaption(ax, sprintf("(%s) %s grain boundary", char('a' + i - 1), sigmaLabels(i)), ...
        fontName, fontSizeCaption);
    hold(ax, "off");
end

putFigureCaption(figB, ...
    "Dynamic stress-strain responses under different hydrogen conditions for each grain boundary", ...
    fontName, 13);
exportgraphics(figB, figB_png, "Resolution", 400);
exportgraphics(figB, figB_pdf, "ContentType", "vector");

disp("Finished.");

%% ===================== Local functions =====================
function key = makeCaseKey(sigma, cond)
    key = matlab.lang.makeValidName(char(string(sigma) + "_" + string(cond)));
end

function formatAxes(ax, xLabelText, yLabelText, fontName, fontSizeAxis, axesLineWidth, xlimManual, ylimManual)
    box(ax, "on");
    grid(ax, "on");
    ax.FontName = fontName;
    ax.FontSize = fontSizeAxis;
    ax.LineWidth = axesLineWidth;
    xlabel(ax, xLabelText, "FontName", fontName);
    ylabel(ax, yLabelText, "FontName", fontName);
    if ~isempty(xlimManual); xlim(ax, xlimManual); end
    if ~isempty(ylimManual); ylim(ax, ylimManual); end
end

function putSubcaption(ax, txt, fontName, fontSizeCaption)
    text(ax, 0.5, -0.18, txt, "Units", "normalized", ...
        "HorizontalAlignment", "center", "VerticalAlignment", "top", ...
        "FontName", fontName, "FontSize", fontSizeCaption, "Interpreter", "tex");
end

function putFigureCaption(fig, txt, fontName, fontSizeCaption)
    annotation(fig, "textbox", [0.05 0.005 0.90 0.035], ...
        "String", txt, "EdgeColor", "none", "HorizontalAlignment", "center", ...
        "VerticalAlignment", "middle", "FontName", fontName, ...
        "FontSize", fontSizeCaption, "Interpreter", "none");
end
