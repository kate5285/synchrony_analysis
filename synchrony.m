clear all; close all; %clc
%%

date={'230111/','230110/','230109/','230108/','230107/','230106/'};
mice='MU31_1/';
% date={'230111/','230110/','230109/','230107/','230106/'};
% mice='MU31_2/';
% in the case of mouse 1, please select date 230108, in case of 2 don't select.
% the codes below are run only on the first loading 
% allExpstruct = {};
% allFczdata = {};
% allpsth= {};
% allpsthrmarttimeline={};
% allvalid={};%is this necessary
% allofflineholoreq={};%is this necessary
% allRavgholormart={};
% allvis={};
% allRall={};

    Ravgholormart_mouse=cell(numel(date), 1);
    expstruct_mouse = cell(numel(date), 1);
    fczdata_mouse = cell(numel(date), 1);
    psth_mouse = cell(numel(date), 1);
    psthrmarttimeline_mouse = cell(numel(date), 1);
    valid_mouse = cell(numel(date), 1);
    offlineholoreq_mouse = cell(numel(date), 1);
    vis_mouse= cell(numel(date), 1);
    Rall_mouse=cell(numel(date), 1);
    %%
for i = 1:numel(date)%%%% MU31_1의 230110일 경우 SGholo2로 다 들여야함. 
pathpp = ['//shinlab/ShinLab/MesoHoloExpts/mesoholoexpts_postprocessed/' mice date{i}];

load([pathpp 'offlineSGholo_rmart.mat'],'Ravgholormart','Rholormart')
% load([pathpp 'offlineSGholo_psth_rmart.mat'])
load([pathpp 'offlineSGholo.mat'], 'validtargets','SGholoexptidn')
load([pathpp 'postprocessed_psth.mat'],'psthall')
load([pathpp 'postprocessed.mat'],'vis','Rall')
if strcmp(mice, 'MU31_1/') && strcmp(date{i}, '230110/')
    load([pathpp 'rmartifact_SGholo2.mat'])
else
load([pathpp 'rmartifact_SGholo.mat'])
end

holodaqpath= '//shinlab/ShinLab/MesoHoloExpts/mesoholoexpts_scanimage/TempHoloOutfiles/';
holodaqfolder= date{i};
if strcmp(mice, 'MU31_1/') && strcmp(date{i}, '230110/')
    mousedata = strrep(mice, '/', '_SGholo2/');
    filename = [strrep(date{i}, '/', ['_', mice(1:end-1), '_','SGholo2_A']) '.mat'];
else
mousedata = strrep(mice, '/', '_SGholo/');
filename = [strrep(date{i}, '/', ['_', mice(1:end-1), '_','SGholo_A']) '.mat'];
end
load([holodaqpath holodaqfolder mousedata filename])
psthrmarttimeline_mouse{i} = (0:size(psthrmart.Fczcell,3)-1)*mean(diff(psthall.("SGholo_x").psthtimeline));%what is ICholo_merged?
expstruct_mouse{i} = ExpStruct;
valid_mouse{i} = validtargets;
vis_mouse{i}=vis;
Rall_mouse{i}=Rall;
Ravgholormart_mouse{i}=Ravgholormart;
fczdata_mouse{i}.Fczcell = Rholormart.Fczcell;
fczdata_mouse{i}.artlines0_Fczcell = Rholormart.artlines0_Fczcell;

psth_mouse{i}.Fczcell = psthrmart.Fczcell;
psth_mouse{i}.artlines0_Fczcell = psthrmart.artlines0_Fczcell;
end

for i = 1:numel(date)
if strcmp(mice, 'MU31_1/') && strcmp(date{i}, '230110/')
    SG ='SGholo2_x';
    % filename = [strrep(date{i}, '/', ['_', mice(1:end-1), '_','SGholo2_A']) '.mat'];
else
SG =SGholoexptidn;
end
pathpp = ['//shinlab/ShinLab/MesoHoloExpts/mesoholoexpts_postprocessed/' mice date{i}];
load(sprintf('%sofflineHoloRequest_%s.mat',pathpp,SG))
offlineholoreq_mouse{i} = offlineHoloRequest;
end

    %% run unitl here and then go back to load other mouses
    allExpstruct = [allExpstruct; expstruct_mouse];
    allFczdata = [allFczdata; fczdata_mouse];
    allpsth = [allpsth; psth_mouse];
    allpsthrmarttimeline = [allpsthrmarttimeline; psthrmarttimeline_mouse];
    allvalid = [allvalid; valid_mouse];
    allofflineholoreq = [allofflineholoreq; offlineholoreq_mouse];
    allRavgholormart=[allRavgholormart;Ravgholormart_mouse];
    allvis=[allvis;vis_mouse];
    allRall=[allRall;Rall_mouse];
%% histogram
asynchronous_data = [];
synchronous_data = [];
auroc_values = [];
neuronums=[];
neuronuma=[];
labels=[];

%find holo conditions (orientation)
Nholoconds = cell(11, 1);
for k = 1:11 
holocondinds = find(allExpstruct{k,1}.holoStimParams.powers > 0);
Nholoconds{k} = length(holocondinds);
end
%모든 cell에 대해 AUROC를 먼저 구하고, 해당되는 뉴런들만 추출
for iholo = 1:2:Nholoconds{k}%need to fix
    neuronuma=[];
    AUCall=[];
for k = 1:11
    Nvaltrials = size(allFczdata{k,1}.Fczcell, 2);
    ilabel = ceil(iholo/length(allExpstruct{k,1}.holoRequest.multiplexgroups));
    neuoind = allofflineholoreq{k,1}.targetneurons(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==ilabel);
    trialsoi_sync = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo);
    trialsoi_async = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo+1);

AUROCICholo =[];
PmwwICholo =[];

data_async =mean(allFczdata{k,1}.Fczcell(neuoind, trialsoi_async),2);%should I do mean->교수님 옛날 코드에 따라 trialsoi로 평균 안 함
data_sync = mean(allFczdata{k,1}.Fczcell(neuoind, trialsoi_sync),2);
labels_async = ones(size(data_async));
labels_sync = zeros(size(data_sync));
labels = [labels_async; labels_sync];

% % perfcurve로 AUROC 계산
[~, ~, ~, AUC] = perfcurve(labels, [data_async; data_sync], 1);
AUCall=[AUCall;AUC];

neuronu=numel(neuoind);
neuronuma=[neuronuma;neuronu];

end
binEdges = 0:0.05:1;
    counts = histcounts(AUCall, binEdges);
    neuron_counts = zeros(size(counts));
    
    for i = 1:length(counts)
        indices = find(AUCall >= binEdges(i) & AUCall < binEdges(i+1));
        neuron_counts(i) = sum(neuronuma(indices));
    end
    
    % 히스토그램 스타일로 그리기
    % figure;
    subplot(3, 2, ilabel);
    bar(binEdges(1:end-1), neuron_counts, 'b');
    xlabel('AUROC');
    ylabel('Neuron Count');
    title('AUROC Neuron Count',name{ilabel});
disp(sum(neuron_counts(:)))

end
%% 방식 2인가요 (전부로 AUROC계산하고 나중에 뽑아내기)
%  (asynchronous는 1, synchronous는 0)
allAUROCiholo=cell(11,1);
allPmwwICholo=cell(11,1);
for k=1:11
Nneurons=size(allFczdata{k,1}.Fczcell,1);
AUROCICholo = NaN(Nneurons, Nholoconds{k}/2);
PmwwICholo = NaN(Nneurons, Nholoconds{k}/2);

for iholo = 1:2:Nholoconds{k}
Nvaltrials = size(allFczdata{k,1}.Fczcell, 2);
ilabel = ceil(iholo/length(allExpstruct{k,1}.holoRequest.multiplexgroups));
trialsoi_sync = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo);
trialsoi_async = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo+1);
for ci = 1:Nneurons
p = ranksum(allFczdata{k,1}.Fczcell(ci,trialsoi_sync), allFczdata{k,1}.Fczcell(ci,trialsoi_async));
PmwwICholo(ci,ilabel) = p;
[X,Y,T,AUC] = perfcurve([ones(1,nnz(trialsoi_async)) zeros(1,nnz(trialsoi_sync))], ...
[allFczdata{k,1}.Fczcell(ci,trialsoi_async) allFczdata{k,1}.Fczcell(ci,trialsoi_sync)], '1');
AUROCICholo(ci,ilabel) = AUC;
end
end
allAUROCiholo{k,1}=AUROCICholo;
allPmwwICholo{k,1}=PmwwICholo;
end
%% 그래프 그리기     
name = {'0 deg', '45 deg', '90 deg', '135 deg','mixed'};
save_path='d:\Users\USER\Documents\MATLAB\AUROC_SGholo';
for ilabel=1:5
    auroc_values=[];
     p_values=[];
    for k=1:11
        neuoind = allofflineholoreq{k,1}.targetneurons(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==ilabel);

        auroc_value=allAUROCiholo{k,1}(neuoind,ilabel);
        auroc_values=[auroc_values;auroc_value];
        p_value=allPmwwICholo{k,1}(neuoind,ilabel);
        p_values=[p_values;p_value];
    end
    % AUROC 값에 해당하는 뉴런 수 히스토그램 그리기
    binEdges = 0:0.05:1;
    counts = histcounts(auroc_values, binEdges);
    neuron_counts = zeros(size(counts));
    signeuron_counts=zeros(size(counts));
    for i = 1:length(counts)
        indices = find(auroc_values >= binEdges(i) & auroc_values < binEdges(i+1));
        significant_indices=find(p_values(indices)<0.05); % significant neurons <0.05
        neuron_counts(i) = numel(indices);
        signeuron_counts(i) = numel(significant_indices);
    end
    disp(sum(neuron_counts(:)))
    % 히스토그램 스타일로 그리기
    subplot(3, 2, ilabel);
    bar(binEdges(1:end-1), neuron_counts, 'b');
    hold on;
    bar(binEdges(1:end-1), signeuron_counts, 'r');
    hold off;

    xlabel('AUROC');
    ylabel('Neuron Count');
    title(sprintf('%s AUROC Neuron Count',name{ilabel}));

end
filename = sprintf('artlines0_Fczcell prof 2 method with significant cells.png');
% saveas(gcf, fullfile(save_path, filename));

%% nonstim 그래프 그리기
name = {'0 deg', '45 deg', '90 deg', '135 deg','mixed'};
save_path='d:\Users\USER\Documents\MATLAB\AUROC_SGholo';
for ilabel=1:5
    auroc_values=[];
     p_values=[];
    for k=1:11
        neuoind = allofflineholoreq{k,1}.targetneurons(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==ilabel);
        nonstimneuoind=setdiff(1:size(allAUROCiholo{k,1}, 1), neuoind);
        auroc_value=allAUROCiholo{k,1}(nonstimneuoind,ilabel);
        auroc_values=[auroc_values;auroc_value];
        p_value=allPmwwICholo{k,1}(nonstimneuoind,ilabel);
        p_values=[p_values;p_value];
    end
    % AUROC 값에 해당하는 뉴런 수 히스토그램 그리기
    binEdges = 0:0.05:1;
    counts = histcounts(auroc_values, binEdges);
    neuron_counts = zeros(size(counts));
    signeuron_counts=zeros(size(counts));
    for i = 1:length(counts)
        indices = find(auroc_values >= binEdges(i) & auroc_values < binEdges(i+1));
        significant_indices=find(p_values(indices)<0.05); % significant neurons <0.05
        neuron_counts(i) = numel(indices);
        signeuron_counts(i) = numel(significant_indices);
    end
    
    % 히스토그램 스타일로 그리기
    subplot(3, 2, ilabel);
    bar(binEdges(1:end-1), neuron_counts, 'b');
    hold on;
    bar(binEdges(1:end-1), signeuron_counts, 'r');
    hold off;

    xlabel('AUROC');
    ylabel('Neuron Count');
    title(sprintf('%s AUROC Neuron Count',name{ilabel}));

end
filename = sprintf(' non stim Fczcell prof 2 method with significant cells.png');
% saveas(gcf, fullfile(save_path, filename));
%% wilcoxon signed rank test

labels_async=[];
labels_sync=[];
%find holo conditions (orientation)
Nholoconds = cell(11, 1);
significant_neurons = []; % significant neurons

for k = 1:11 
holocondinds = find(allExpstruct{k,1}.holoStimParams.powers > 0);
Nholoconds{k} = length(holocondinds);
end

for iholo = 1:2:Nholoconds{k} % need to fix
auroc_values = []; 
labels_async = []; % Asynchronous 그룹의 레이블
labels_sync = []; % Synchronous 그룹의 레이블
neuronuma=[];neuronums=[];
significant_neurons =[];
 data_A=[];
 data_S=[];
    for k = 1:11
        Nvaltrials = size(allFczdata{k,1}.artlines0_Fczcell, 2);
        ilabel = ceil(iholo/length(allExpstruct{k,1}.holoRequest.multiplexgroups));
        neuoind_sync = allofflineholoreq{k,1}.targetneurons(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==ilabel);
        neuoind_async = allofflineholoreq{k,1}.targetneurons(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==ilabel);
        trialsoi_sync=[];%don't think including this and next line is necessary
        trialsoi_async =[];
        trialsoi_sync = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo);
        trialsoi_async = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo+1);

        data_async = mean(allFczdata{k,1}.artlines0_Fczcell(neuoind_async, trialsoi_async), 2);%should I do mean
        data_sync = mean(allFczdata{k,1}.artlines0_Fczcell(neuoind_sync, trialsoi_sync), 2); 
         data_A=[data_A;data_async];
         data_S=[data_S;data_sync];
        labels_async = [ones(size(data_async))];
        labels_sync = [zeros(size(data_sync))];
        labels = [labels_async; labels_sync];

        neuronu=numel(neuoind_sync);
        neuronums=[neuronums;neuronu];

        [~, ~, ~, AUC] = perfcurve(labels, [data_async; data_sync], 1);
        auroc_values = [auroc_values; AUC];
    end
    p_value = ranksum(data_A, data_S);
    disp(['P-value for A vs S values: ' num2str(p_value)]);

    % 유의미한 뉴런 수 계산
    % [p_value, ~, stats] = signrank(auroc_values ,0.5);
    % disp(['P-value for AUROC values vs 0.5: ' num2str(p_value)]);
    
    % p-value가 0.05보다 작으면 빨간색으로 표시
    if p_value < 0.05
       significant_neurons = [significant_neurons; neuronums];
    end

    % AUROC 값에 해당하는 뉴런 수 히스토그램 그리기
    binEdges = 0:0.05:1;
    counts = histcounts(auroc_values, binEdges);
    neuron_counts = zeros(size(counts));

    for i = 1:length(counts)
        indices = find(auroc_values >= binEdges(i) & auroc_values < binEdges(i+1));
        neuron_counts(i) = sum(neuronums(indices));
    end

    % 히스토그램 스타일로 그리기
    red_indices = find(ismember(neuronums, significant_neurons));
    figure;
    bar(binEdges(1:end-1), neuron_counts, 'b');
    hold on;
    bar(binEdges(red_indices), neuron_counts(red_indices), 'r');
    xlabel('AUROC');
    ylabel('Significant Neuron Count');
    title('Significant Neuron Count based on AUROC');
 red_indices=[];
end

%% psth heatmap
%find holo conditions (orientation)
psth_a=[];
psth_s=[];
psth_a_accumulated = [];
psth_s_accumulated = [];
Nholoconds = cell(11, 1);
for k = 1:11 
holocondinds = find(allExpstruct{k,1}.holoStimParams.powers > 0);
Nholoconds{k} = length(holocondinds);
end

for iholo = 1:2:Nholoconds{k}
for k = 1:11
        Nvaltrials = size(allFczdata{k,1}.artlines0_Fczcell, 2);
        ilabel = ceil(iholo/length(allExpstruct{k,1}.holoRequest.multiplexgroups));
        neuoind_sync = allofflineholoreq{k,1}.targetneurons(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==ilabel);
        trialsoi_sync = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo);
        trialsoi_async = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo+1);
       psth_async = squeeze(mean(allpsth{k,1}.Fczcell(neuoind_sync,trialsoi_async,:), 2));%should I do mean
       psth_sync = squeeze(mean(allpsth{k,1}.Fczcell(neuoind_sync, trialsoi_sync,:), 2));
       psth_a=[psth_a;psth_async];
       psth_s=[psth_s;psth_sync];    
end
[~, async_order] = sort(mean(psth_a, 2), 'descend');
    [~, sync_order] = sort(mean(psth_s, 2), 'descend');
    
    % 정렬된 순서대로 누적된 배열 쌓기
    psth_a_accumulated = [psth_a_accumulated; psth_a(async_order, :)];
    psth_s_accumulated = [psth_s_accumulated; psth_s(sync_order, :)];
figure;
% Plot PSTH
        subplot(1, 2, 1);
        imagesc(psth_a_accumulated);
        title('PSTH Asynchronous');
        xlabel('Time');
        ylabel('Neuron');
        colorbar;
        caxis([-1 3.5]);
        set(gca, 'Position', [0.05, 0.1, 0.3, 0.8]); % 좌측 subplot 위치

        subplot(1, 2, 2);
        imagesc(psth_s_accumulated);
        title('PSTH Synchronous');
        xlabel('Time');
        ylabel('Neuron');
        colorbar;
        caxis([-1 3.5]);
        set(gca, 'Position', [0.55, 0.1, 0.3, 0.8]); % 우측 subplot 위치
        psth_a=[];
        psth_s=[];
        psth_a_accumulated =[];
        psth_s_accumulated =[];
end

%% load off and online data 
% for each ONLINE ROI, find the OFFLINE ROI that is nearest.
% allneuronXYcoords=[];
% alliscell=[];
% alloffline=[];
% allonlineSGHR=[];
% date={'230111/','230110/','230109/','230108/','230107/','230106/'};
% mice='MU31_1/';
date={'230111/','230110/','230109/','230107/','230106/'};
mice='MU31_2/';

neuronXYcoords_mouse=cell(numel(date), 1);
iscell_mouse=cell(numel(date),1);
offline_mouse=cell(numel(date),1);
onlineSGHR_mouse=cell(numel(date),1);
for i = 1:numel(date)
onlinepath = ['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts_scanimage\' mice date{i} '\ClosedLoop_justgreen\'];
load([onlinepath 'online_params.mat'],'neuronXYcoords')
[SGholoreqfile,SGholoreqpath] = uigetfile([onlinepath '*.mat'], 'choose SG holoRequest file from onlinepath');
onlineSGHR = load([SGholoreqpath SGholoreqfile]);
neuronXYcoords_mouse{i}=neuronXYcoords;
load(['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts_postprocessed\' mice date{i} '\postsuite2p_params.mat'])
iscell_mouse{i}=iscell;
offlinepath = ['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts\' mice date{i} '\SGholo\'];
offline = load([offlinepath 'Fall_splitx.mat']);
offline_mouse{i}=offline;
onlineSGHR_mouse{i}=onlineSGHR;
end

%%
allneuronXYcoords = [allneuronXYcoords; neuronXYcoords_mouse];
alliscell=[alliscell;iscell_mouse];
alloffline=[alloffline;offline_mouse];
allonlineSGHR=[allonlineSGHR;onlineSGHR_mouse];
%%
all_Noffrois=cell(11,1);%number of all dates and mouse
all_imoffroi=cell(11,1);
all_offroictr=cell(11,1);
for i =1:11
all_Noffrois{i,1} = numel(alloffline{i,1}.stat);
all_imoffroi{i,1} = zeros(alloffline{i,1}.ops.Ly, alloffline{i,1}.ops.Lx);
all_offroictr{i,1} = zeros(all_Noffrois{i,1},2);
end

for i =1:11
for ci = 1:all_Noffrois{i,1}
tempiminds = sub2ind(size(all_imoffroi{i,1}),alloffline{i,1}.stat{ci}.ypix, alloffline{i,1}.stat{ci}.xpix);
all_imoffroi{i,1}(tempiminds) = ci;
all_offroictr{i,1}(ci,:)=double(alloffline{i,1}.stat{ci}.med);
end
end

cropedgethr = 50;
fname = "\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts_scanimage\MU31_2\230111\ICholo\file_00001.tif";

header = imfinfo(fname);
artist_info     = header(1).Artist;
% % retrieve ScanImage ROIs information from json-encoded string 
artist_info = artist_info(1:find(artist_info == '}', 1, 'last'));
artist = jsondecode(artist_info);
hSIh = header(1).Software;
hSIh = regexp(splitlines(hSIh), ' = ', 'split');
for n=1:length(hSIh)
	if strfind(hSIh{n}{1}, 'SI.hRoiManager.scanVolumeRate')
		fs = str2double(hSIh{n}{2});
	end
end
si_rois = artist.RoiGroups.imagingRoiGroup.rois;
nrois = numel(si_rois);
Ly = [];
Lx = [];

for k = 1:nrois
	Ly(k,1) = si_rois(k).scanfields(1).pixelResolutionXY(2);
	Lx(k,1) = si_rois(k).scanfields(1).pixelResolutionXY(1);
end

%validoffcells??
% validoffXYcells = min(abs(offroictr(:,2)-cumsum([0 Lx'])),[],2)>cropedgethr & ...
%     min(abs(offroictr(:,1)-[0 unique(Ly')]),[],2)>cropedgethr;

all_offXYcoords = all_offroictr;
%offroipairumdist = sqrt( (yumperpix*(offroictr(:,1)-offroictr(:,1)')).^2 + (xumperpix*(offroictr(:,2)-offroictr(:,2)')).^2 );
% onoffroiumdist = sqrt((yumperpix*(allExpstruct{7,1}.holoRequest.targets(:,1)-offXYcoords(:,1)')).^2 + ...
%     (xumperpix*(allExpstruct{7,1}.holoRequest.targets(:,2)-offXYcoords(:,2)')).^2);
all_targetROIdist=cell(11,1);
alltargetroiXYdist=cell(11,1);
% x/yumperpix가 trial마다 동일하다 가정함
load(['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts_scanimage\MU31_2\230111\ClosedLoop_justgreen\holoRequest_MU31_2_230111_staticgratings_002.mat'],'fullxsize_orig','fullysize_orig')
xumperpix = fullxsize_orig/size(offline.ops.meanImg,2);
yumperpix = fullysize_orig/size(offline.ops.meanImg,1);
%%
for i =1:11
% target과 ROI간의 거리
all_targetROIdist{i,1} = sqrt((yumperpix*(allofflineholoreq{i,1}.targets(:,1)-all_offXYcoords{i,1}(:,1)')).^2 + ...
    (xumperpix*(allofflineholoreq{i,1}.targets(:,2)-all_offXYcoords{i,1}(:,2)')).^2);
% alltargetroiXYdist{i,1} = all_targetROIdist{i,1}(allonlineSGHR{i,1}.ontarginds, alliscell{i,1});
alltargetroiXYdist{i,1} = all_targetROIdist{i,1}(:, alliscell{i,1});
end
%% activity and distance plot
name = {'0 deg', '45 deg', '90 deg', '135 deg','mixed'};
save_path='d:\Users\USER\Documents\MATLAB\AUROC_SGholo\binned distance and reactivity plot+ error bars';
screenSize = get(0, 'ScreenSize');
figure('Position', screenSize);

for k = 1:11 
holocondinds = find(allExpstruct{k,1}.holoStimParams.powers > 0);
Nholoconds{k} = length(holocondinds);
end

for ilabel= 1:ceil(Nholoconds{k}/length(allExpstruct{k,1}.holoRequest.multiplexgroups))
mindist_all=[];
s_activity_all=[];
a_activity_all=[];

for k= 1:11
labeleddist=alltargetroiXYdist{k,1}(allofflineholoreq{k,1}.holoGroups'==ilabel,:);
r = zeros(1,size(labeleddist, 2));
mindist=  min(labeleddist,[],1);
durholoinds = 12:16;
Rdurholo = squeeze(nanmean(allpsth{k,1}.artlines0_Fczcell(:,:,durholoinds), 3));
% preholoinds = 1:5; % 1-5
% Rpreholo = squeeze(nanmean(allpsth{k,1}.Fczcell(:,:,preholoinds), 3));
% tempRholo = Rdurholo(neu2include,:);
% tempRbase = Rpreholo(neu2include,:);

for i = 1:size(labeleddist,2) %같은 orientation을 가진 타겟(뉴런들) 묶음
% [~, r(i)] = find(labeleddist(:, i) == mindist(i));
indices = find(labeleddist(:, i) == mindist(i));
r(i) = indices(1);%동일한 거리를 가지는 경우 첫 번째를 부르게 함. 확인 필요!!
% //그리고 r은 사용하지는 않고 수십개의 타겟 뉴런 중 몇번째가 가장 가까운지 넣은 인덱스임.(확인용)
end

Nvaltrials = size(allFczdata{k,1}.artlines0_Fczcell, 2);
trialsoi_sync = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(2*ilabel-1);
trialsoi_async = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(2*ilabel);
% a_activity= mean(allFczdata{k,1}.artlines0_Fczcell(:,trialsoi_async),2);
% s_activity= mean(allFczdata{k,1}.artlines0_Fczcell(:,trialsoi_sync),2);
a_activity= mean(Rdurholo(:,trialsoi_async),2);%Fczdata와 동일한 time period
s_activity= mean(Rdurholo(:,trialsoi_sync),2);
s_activity_all=[s_activity_all;s_activity];
a_activity_all=[a_activity_all;a_activity];
mindist_all=horzcat(mindist_all,mindist);
end

% binSize = 10; %linear bin 
% edges = 0:binSize:(max(mindist_all)+binSize); 
% [~, ~,binIdx] = histcounts(mindist_all, edges);
% 
% sbinMeans = accumarray(double(binIdx(:)), s_activity_all(:), [], @mean);
% sbinStd = accumarray(double(binIdx(:)), s_activity_all(:), [], @std);%standard deviation
% sbinSE = accumarray(binIdx(:), s_activity_all(:), [], @(x) std(x) / sqrt(numel(x)));
% 
% abinMeans = accumarray(double(binIdx(:)), a_activity_all(:), [], @mean);
% abinStd = accumarray(double(binIdx(:)), a_activity_all(:), [], @std);%standard deviation
% abinSE = accumarray(binIdx(:), a_activity_all(:), [], @(x) std(x) / sqrt(numel(x)));
% 
% confidenceLevel = 0.95;
% z = norminv(1 - (1 - confidenceLevel) / 2);
% sbinCI = sbinSE * z;%confidence level
% abinCI = abinSE * z;
% 
% subplot(3, 2, ilabel);
% errorbar(edges(1:end-1)+binSize/2, sbinMeans, sbinSE, 'k-','Marker', '.', 'LineWidth', 1.5);
% % plot(mindist_all, s_activity_all,'.') % for now, considered iscell so I used alltarget~. should I use this?
% legend('sync')
% hold on;
% % plot(mindist_all, a_activity_all,'.')
% errorbar(edges(1:end-1)+binSize/2, abinMeans, abinSE, 'r-','Marker', '.', 'LineWidth', 1);
% hLine=yline(0, 'b-', 'LineWidth', 1);
% legend(hLine, 'Hide in Legend', 'AutoUpdate', 'off');
% legend({'sync','async'},'Location', 'northeast')
% ylabel('Response')
% xlim([0,800]);
% xticks(0:100:max(edges));
% xlabel('distance from target (\mum) ')
% title(sprintf('%s trialtype',name{ilabel}));

%log-scaled bins. 이 경우 내가 가진 데이터셋 중에서 1000을 넘는 것은 나오지 않음.pref는 나옴
binSize = 0.25; 
minlog=floor(log10(min(mindist_all)));
maxlog=ceil(log10(max(mindist_all)));
if min(mindist_all) == 0
edges = [0 logspace(binSize, maxlog, maxlog/binSize)];
maxDistIndex = find(edges> max(mindist_all(:)),1);
edges = edges(1:maxDistIndex);
else
edges11 = logspace(minlog, maxlog, ceil((maxlog)/binSize) + 1);
minDistIndex = find(edges11> min(mindist_all(:)),1);
minDistRange = edges11(minDistIndex-1:end);
maxDistIndex = find(edges11> max(mindist_all(:)),1);
edges = edges11(minDistIndex-1:maxDistIndex);
end
[~, ~, binIdx] = histcounts(mindist_all, edges);

confidenceLevel = 0.95;
z = norminv(1 - (1 - confidenceLevel) / 2);
sbinMeans = accumarray(binIdx(:), s_activity_all(:), [], @mean);
sbinStd = accumarray(double(binIdx(:)), s_activity_all(:), [], @std);
sbinSE = accumarray(binIdx(:), s_activity_all(:), [], @(x) std(x) / sqrt(numel(x)));
sbinCI = sbinStd * z;

abinMeans = accumarray(binIdx(:), a_activity_all(:), [], @mean); 
abinStd = accumarray(double(binIdx(:)), a_activity_all(:), [], @std);
abinSE = accumarray(binIdx(:), a_activity_all(:), [], @(x) std(x) / sqrt(numel(x)));
abinCI = abinStd * z;

subplot(3, 2, ilabel);
errorbar(edges(1:end-1)+binSize/2, sbinMeans, sbinSE, 'k-','Marker', '.', 'LineWidth', 1.5);
legend('sync')
hold on;
errorbar(edges(1:end-1)+binSize/2, abinMeans, abinSE, 'r-','Marker', '.', 'LineWidth', 1);
ylabel('Response')
hLine=yline(0, 'b-', 'LineWidth', 1);
legend(hLine, 'Hide in Legend', 'AutoUpdate', 'off');
legend({'sync','async'},'Location', 'northeast')
xlabel('Distance from target (\mum) (logscale bin)')
set(gca,'XScale','log'); % Set x-axis to log scale

end
filename = sprintf('yyall log bin different time artlines0_Fczcell activity and distance plot standard error.png');
saveas(gcf, fullfile(save_path, filename));
%% nonstim
% or I can use target2celldist;no this is different.min이어서 안되는 것인듯.

asynchronous_data = [];
synchronous_data = [];
auroc_values = [];
neuronums=[];
neuronuma=[];
labels=[];
%find holo conditions (orientation)
Nholoconds = cell(11, 1);
for k = 1:11 
holocondinds = find(allExpstruct{k,1}.holoStimParams.powers > 0);
Nholoconds{k} = length(holocondinds);
end

for iholo = 1:2:Nholoconds{1}
    offtarget =[];
    for k = 1:11
        Nvaltrials = size(allFczdata{k,1}.artlines0_Fczcell, 2);
        ilabel = ceil(iholo/length(allExpstruct{k,1}.holoRequest.multiplexgroups));
        [R, ~] = find(allofflineholoreq{k,1}.holoGroups'==ilabel);
        selected_rows = alltargetroiXYdist{k,1}(R, :);
        [r, c] = find(selected_rows > 25);
        rc = [R(r), c];
        % Q// 여기서 타겟xROI 배열인데 어떻게 Fczcell안으로 인덱싱을 하지..
        trialsoi_sync = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo);
        trialsoi_async = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo+1);
        % neuoind= allofflineholoreq{k,1}.targetneurons(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==ilabel);%neuoind는 

        ROIS = zeros(numel(unique(R)) * numel(unique(c)), 2);
        ROIS(:, 1) = reshape(repmat(R, 1, numel(unique(c))), [], 1);
        ROIS(:, 2) = reshape(repmat(1:numel(unique(c)), numel(unique(R)), 1), [], 1);
        missing_values = setdiff(ROIS, rc, 'rows');
        nonstimROI=setdiff(1:numel(unique(c)), unique(missing_values(:, 2)));         
        data_async = mean(allFczdata{k,1}.Fczcell(nonstimROI, trialsoi_async), 2);
        data_sync = mean(allFczdata{k,1}.Fczcell(nonstimROI, trialsoi_sync), 2);

        % 레이블 생성 (asynchronous는 1, synchronous는 0)
        labels_async = ones(size(data_async));
        labels_sync = zeros(size(data_sync));
        labels = [labels_async; labels_sync];
        neuronu=numel(nonstimROI);
        neuronums=[neuronums;neuronu];
        neuronu=numel(nonstimROI);
        neuronuma=[neuronuma;neuronu];

        % perfcurve로 AUROC 계산
        [~, ~, ~, AUC] = perfcurve(labels, [data_async; data_sync], 1);
        auroc_values = [auroc_values; AUC];
    end
    p_value = ranksum(data_A, data_S);
    disp(['P-value for A vs S values: ' num2str(p_value)]);
    % AUROC 값에 해당하는 뉴런 수 히스토그램 그리기
    binEdges = 0:0.05:1;
    counts = histcounts(auroc_values, binEdges);
    neuron_counts = zeros(size(counts));
    
    for i = 1:length(counts)
        indices = find(auroc_values >= binEdges(i) & auroc_values < binEdges(i+1));
        neuron_counts(i) = sum(neuronuma(indices));
    end

    figure;
    bar(binEdges(1:end-1), neuron_counts, 'b');
    xlabel('AUROC');
    ylabel('Neuron Count');
    title('AUROC Neuron Count');

    neuronuma=[];neuronums=[];auroc_values=[];

end
%%
Nholoconds=10;%need to fix
for iholo = 1:2:Nholoconds
    dataA=[];
    dataS=[];
    dist=[];
for k = 1:11
        Nvaltrials = size(allFczdata{k,1}.artlines0_Fczcell, 2);
        ilabel = ceil(iholo/length(allExpstruct{k,1}.holoRequest.multiplexgroups));
        nonstim_indices = find(allofflineholoreq{k, 1}.target2celldist(:, 1) > 25);
        [R, ~] = find(allofflineholoreq{k,1}.holoGroups'==ilabel);
        selected_rows = alltargetroiXYdist{k,1}(R, :);
        [r, c] = find(selected_rows > 25);
        rc = [R(r), c];
        ROIS = zeros(numel(unique(R)) * numel(unique(c)), 2);
        ROIS(:, 1) = reshape(repmat(R, 1, numel(unique(c))), [], 1);
        ROIS(:, 2) = reshape(repmat(1:numel(unique(c)), numel(unique(R)), 1), [], 1);
        missing_values = setdiff(ROIS, rc, 'rows');
        nonstimROI=setdiff(1:numel(unique(c)), unique(missing_values(:, 2)));     

        trialsoi_sync = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo);
        trialsoi_async = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo+1);
        distances =mean(alltargetroiXYdist{k,1}(R,nonstimROI),1); %Q// distance.should I do min? there are multiple targets for orientation
        data_async = mean(allFczdata{k,1}.Fczcell(nonstimROI, trialsoi_async), 2);%should I do mean
        data_sync = mean(allFczdata{k,1}.Fczcell(nonstimROI, trialsoi_sync), 2);
        dataA=[dataA;data_async];
        dataS=[dataS;data_sync];

        dist=horzcat(dist, distances);
end
figure 

    plot(dist.', dataA, 'b.'); % Blue for async
    hold on;
    plot(dist.', dataS, 'r.'); % Red for sync

    hold off;
end

%% preferred orientation. N ori x N neu 한 neu에 대해 무슨 ori가 max값을 보이는지 확인
%아무리 생각해도 전체 250에서 max찾는 거보다 orientation종류마다 분류해서 각각 평균내고 가장 큰 값을 가질 때의
%orientation을 구하는 게 맞음.
%그리고 Rall을 사용하는 게 맞는지 확인- ROI수가 하나의 축이되는건지, neuoind로 뽑은 게 맞는건지.
%원랜 ROI에 대해 다 돌리는 식으로 했는데, neuoind로  한정하니까 숫자가 안 맞음.
trialorder=[];
Fcz_data=[];
responses_t=[];
one=[];
one1=[];

orientation_responses=cell(5,1);
mean_responses=cell(5,1);
for o = 1:5 %trial orientation 종류대로 입력. 
responses_t=[];
    for k=1:11
    Fcz_data=allRall{k,1}.staticgratings_002.Fcz;%staticgratings가 맞을까요..SGholo아닌가. 이 둘은 trial type가 다른 거 같음-allvis에 없다.
    trialorder=allvis{k,1}.staticgratings_002.trialorder;
    
    one1=[];
    one=[];
        for n=1:size(trialorder,1)
        one1= mod(trialorder(n,1), 10);
        one=[one;one1];
        end
        trialori= one==(o-1);
        % o가 1일때 one의 0,, 이런 식으로 o가 5일 때 one이 4가 되는 roi 인덱스 찾기. one에는 0부터
        % 4까지 있으므로
        % neuoind = allofflineholoreq{k,1}.targetneurons(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==label);
    
        response = Fcz_data(:, trialori);
        responses_t=[responses_t;response];
    end
orientation_responses{o,1}=responses_t;%여기에는 회색인 0부터 4까지 있음.총 5개
end

for o=2:5 
mean_responses{o,1}=mean(orientation_responses{o,1},2);%그래서 회색 화면인 0을 제외
end
combined_data = cat(2, mean_responses{:});%여기에 4종류의 preferred orientation이 col으로 들어감.row는 ROI수

[~, max_idx]= max(combined_data,[],2);
% orthogonal// 이대로 바로 쓰지 않았고 아래 3블록 쯤 뒤 코드에서 ortho// trialtype으로 20가지로 나눔
ortho_idx=zeros(size(combined_data, 1), 1);
for k = 1:size(combined_data, 1)
    %  일의 자릿수 기준으로 수직 orientation 계산
    ortho_idx(k,1) = mod(max_idx(k,1) + 1, 4) + 1;
end

%% 데이터 추출
ROIsizes=[];
max_idxi=cell(11,1);
combined_pertrial=cell(11,1);
for k=1:11
ROIsize=size(allRall{k,1}.staticgratings_002.Fcz,1);
ROIsizes=[ROIsizes;ROIsize];
if k==1
max_idxi{k,1}=max_idx(1:ROIsizes);
combined_pertrial{k,1}=combined_data(1:ROIsizes,:);
else
max_idxi{k,1}=max_idx(sum(ROIsizes(1:k-1))+1:sum(ROIsizes(1:k)));
combined_pertrial{k,1}=combined_data(sum(ROIsizes(1:k-1))+1:sum(ROIsizes(1:k)),:);
end
end
dist_ori_all=cell(4,1);
data_async_ori_all=cell(4,1);% cell 크기는 pref의 가짓수
data_sync_ori_all=cell(4,1);
neuoind_both_ori_all=cell(4,1);
neuoind_both_oi=cell(5,1);
data_async_ori=cell(5,1);% cell 크기는 sitm의 가짓수
data_sync_ori=cell(5,1);
dist_ori=cell(5,1);
data_async_o=cell(11,1);
data_sync_o=cell(11,1);
dist_o=cell(11,1);

Nholoconds = cell(11, 1);
for k = 1:11 
holocondinds = find(allExpstruct{k,1}.holoStimParams.powers > 0);
Nholoconds{k} = length(holocondinds);
end

for o=1:4
for ilabel= 1:ceil(Nholoconds{k}/length(allExpstruct{k,1}.holoRequest.multiplexgroups))%1~5. 5종의 trialtype
         data_async_o=[];
         data_sync_o=[];
         dist_o=[];
         neuoind_both_o=[];
        for k = 1:11
        % neuoind= allofflineholoreq{k,1}.targetneurons(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==ilabel);%neuoind는 
        % 타겟 자체를 찾는 부분 같음 그래서 일단은 거리에 따른 거니까 포함 안할 것..
        %max_idx에서 ori에 맞는 거 추출
        max_ROI = find(max_idxi{k,1} == o);
        Nvaltrials = size(allFczdata{k,1}.Fczcell, 2);
        trialsoi_sync = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(2*ilabel-1);
        trialsoi_async = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(2*ilabel);
        % neuoind_both=neuoind(find(arrayfun(@(x) any(max_ROI == x), neuoind(:))));%stimul과 preference 둘 다를 만족하는 경우
        neuoind_both=max_ROI;

        % durholoinds = 12:16;
        % Rdurholo = squeeze(nanmean(allpsth{k,1}.artlines0_Fczcell(:,:,durholoinds), 3));
        data_async = mean(allFczdata{k,1}.Fczcell(neuoind_both, trialsoi_async), 2);%Q!! should I do mean
        data_sync = mean(allFczdata{k,1}.Fczcell(neuoind_both, trialsoi_sync), 2); 
        % data_async =mean(Rdurholo(neuoind_both,trialsoi_async),2);
        % data_sync =mean(Rdurholo(neuoind_both,trialsoi_sync),2);

        data_async_o=[data_async_o;data_async];
        data_sync_o=[data_sync_o;data_sync];
        % data_async_o{k,1}=data_async;
        % data_sync_o{k,1}=data_sync;

        labeldist=(alltargetroiXYdist{k,1}(allvalid{k,1} & allofflineholoreq{k,1}.holoGroups'==ilabel,neuoind_both));
        dist=min(labeldist,[],1);%Q// 2번째 차원이 모든 ROI에 해당하는 것이고, 1번째 차원은 그 label에 해당되는 타겟들임.
        dist_o = horzcat(dist_o, dist);
        % dist_o{k,1}=dist;
        neuoind_both_o=[neuoind_both_o;neuoind_both];
        end
        data_async_ori{ilabel,1}=data_async_o;
        data_sync_ori{ilabel,1}=data_sync_o;
        dist_ori{ilabel,1}=dist_o';
        neuoind_both_oi{ilabel,1}=neuoind_both_o;
end
data_async_ori_all{o,1}=data_async_ori;
data_sync_ori_all{o,1}=data_sync_ori;
dist_ori_all{o,1}=dist_ori;
neuoind_both_ori_all{o,1}=neuoind_both_oi;
end
%% orientation tuning curve plot
orientations = [0, 45, 90, 135];
datasperori=cell(4,1);

for o=1:4
    row_datas=[];
    for i = 1:11
    % max activity를 보이는 orientation에 대한 ROI 추출
    max_ROI = find(max_idxi{i,1} == o);  % 가장 큰 값의 index

    % 해당되는 row 추출
    row_data = combined_pertrial{i,1}(max_ROI, :);
    row_datas=[row_datas;row_data];
    datasperori{o,1}=row_datas;
    end
end

smoothed_data=cell(4,1);
for o = 1:4
    subplot(2, 2, o);
     smoothed_data{o,1}=[];
    for i = 1:4
        mean_data = mean(datasperori{o, 1}(:,i), 1);
        smoothed_d = smoothdata(mean_data, 'lowess');
         smoothed_data{o,1}= [smoothed_data{o,1}; smoothed_d];
       end
    % 현재 subplot에 대한 hold off
    plot(orientations, smoothed_data{o,1}, '-', 'LineWidth', 2);  % smoothing된 곡선 표시
    title(['Orientation ', num2str(orientations(o))]);
    xlabel('Orientation (degrees)');
    ylabel('Activity');
    grid on;
    xticks(orientations);
end

% 전체 그래프 제목 추가
% suptitle('Orientation Tuning Curves');

%% 그래프 그리기-activity and distance plot
oname = {'0 deg', '45 deg', '90 deg', '135 deg'};
ilabelname={'0 deg', '45 deg', '90 deg', '135 deg','mixed'};
save_path='d:\Users\USER\Documents\MATLAB\AUROC_SGholo\preferred orientation\';

screenSize = get(0, 'ScreenSize');
figure('Position', screenSize);
for o=1:4
    for ilabel=1:5
          
binSize = 10;
maxLinear=max(dist_ori_all{o,1}{ilabel,1}(:));%linear한 경우 0으로 시작해도 문제되지 않고 1000을 넘어도 범주가 max를 뛰어넘어 급증하지 않음
oridist=dist_ori_all{o,1}{ilabel,1}(:);
edges = 0:binSize:maxLinear+binSize;
[~, ~,binIdx] = histcounts(oridist, edges);

sbinMeans = accumarray(double(binIdx(:)), data_sync_ori_all{o,1}{ilabel,1}(:), [], @mean);
sbinStd = accumarray(double(binIdx(:)), data_sync_ori_all{o,1}{ilabel,1}(:), [], @std);%standard deviation
sbinSE = accumarray(binIdx(:), data_sync_ori_all{o,1}{ilabel,1}(:), [], @(x) std(x) / sqrt(numel(x)));

abinMeans = accumarray(double(binIdx(:)), data_async_ori_all{o,1}{ilabel,1}(:), [], @mean);
abinStd = accumarray(double(binIdx(:)), data_async_ori_all{o,1}{ilabel,1}(:), [], @std);%standard deviation
abinSE = accumarray(binIdx(:), data_async_ori_all{o,1}{ilabel,1}(:), [], @(x) std(x) / sqrt(numel(x)));

confidenceLevel = 0.95;
z = norminv(1 - (1 - confidenceLevel) / 2);
sbinCI = sbinSE * z;%confidence level
abinCI = abinSE * z;

subplot(5, 4, (ilabel-1)*4 + o);
errorbar(edges(1:end-1) + binSize/2, sbinMeans, sbinSE, 'k-','Marker', '.','LineWidth', 1.5);
hold on;
errorbar(edges(1:end-1) + binSize/2, abinMeans, abinSE, 'r-','Marker', '.','LineWidth', 1);
hLine=yline(0, 'b-', 'LineWidth', 1.5);
legend(hLine, 'Hide in Legend', 'AutoUpdate', 'off');
legend({'sync','async'},'Location', 'northeast')
ylabel('Response')
yticks(-0.4:0.05:0.35);
ylim([-0.4,0.4]);
xlim([0 800]);
xticks(0:100:max(edges));
xlabel('distance from target (\mum) ')
title(sprintf('%s preferred %s trialtype',oname{o},ilabelname{ilabel}));
      

% binSize = 0.45; %for log scale
% %in this case up to 1000 is fine. problem is after that it jumps multiple steps to 10000(만까지 안 넘어간다는 가정 하)
% maxLog = ceil(log10(1000));
% minLog = floor(log10(min(dist_ori_all{o,1}{ilabel,1}(:))));
% minLog2 = floor(log10(1000));
% maxLog2 = ceil(log10(max(dist_ori_all{o,1}{ilabel,1}(:))));
% edges2 = logspace(minLog2, maxLog2, ceil((maxLog2)/binSize) + 1);
%     maxDistIndex = find(edges2 <= max(dist_ori_all{o,1}{ilabel,1}(:)), 1);
%     maxDistRange = edges2(maxDistIndex:min(maxDistIndex+1, numel(edges2)));
% 
% if min(dist_ori_all{o,1}{ilabel,1}(:)) == 0 %최소가 0이되는 경우.
%     if max(dist_ori_all{o,1}{ilabel,1}(:))>=1000 %1000을 최대가 넘는 경우 
%         edges1 = [0 logspace(binSize, maxLog, ceil(maxLog/binSize))];
%         edges = [edges1, maxDistRange(2:end)];
%     else
%     edges =[0 logspace(binSize, maxLog2, ceil(maxLog2/binSize))];
%     end
% else
%     edges11 = logspace(minLog, maxLog, ceil((maxLog)/binSize) + 1);%1000을 안 넘으니.
%     minDistIndex = find(edges11 > min(dist_ori_all{o,1}{ilabel,1}(:)), 1);
%     minDistRange = edges11(minDistIndex-1:end);
%     if max(dist_ori_all{o,1}{ilabel,1}(:))>=1000
%     edges = [minDistRange, maxDistRange(2:end)];
%     else
%     edges =  minDistRange;    
%     end
% end
% [~, ~,binIdx] = histcounts(dist_ori_all{o,1}{ilabel,1}(:), edges);
% 
% confidenceLevel = 0.95;
% z = norminv(1 - (1 - confidenceLevel) / 2);
% sbinMeans = accumarray(double(binIdx(:)), data_sync_ori_all{o,1}{ilabel,1}(:), [], @mean);
% sbinStd = accumarray(double(binIdx(:)), data_sync_ori_all{o,1}{ilabel,1}(:), [], @std);%standard deviation
% sbinSE = accumarray(binIdx(:), data_sync_ori_all{o,1}{ilabel,1}(:), [], @(x) std(x) / sqrt(numel(x)));
% 
% abinMeans = accumarray(double(binIdx(:)), data_async_ori_all{o,1}{ilabel,1}(:), [], @mean);
% abinStd = accumarray(double(binIdx(:)), data_async_ori_all{o,1}{ilabel,1}(:), [], @std);%standard deviation
% abinSE = accumarray(binIdx(:), data_async_ori_all{o,1}{ilabel,1}(:), [], @(x) std(x) / sqrt(numel(x)));
% sbinCI = sbinSE * z;%confidence level
% abinCI = abinSE * z;
% 
%  subplot(5, 4, (ilabel-1)*4 + o);
% hold on;
% errorbar(edges(1:end-1)+ binSize/2, sbinMeans, sbinSE, 'k-','Marker', '.','LineWidth', 1.5);
% errorbar(edges(1:end-1)+ binSize/2, abinMeans, abinSE, 'r-','Marker', '.','LineWidth', 1);
% hLine=yline(0, 'b-', 'LineWidth', 1);
% legend(hLine, 'Hide in Legend', 'AutoUpdate', 'off');
% legend({'sync','async'},'Location', 'northeast')
% ylabel('Response')
% set(gca, 'YAxisLocation', 'origin')
% 
% xlabel('Distance from target (\mum) (logscale bin)')
% set(gca,'XScale','log'); % Set x-axis to log scale    
% title(sprintf('%s preferred %s trialtype',oname{o},ilabelname{ilabel}));
    end
end

    filename = sprintf('different time log_pref_trialtype_artlines0_Fczcell_activity_and_distance_plot_standard_error.png');
    saveas(gcf, fullfile(save_path, filename));
%% ranksum 
oname = {'0 deg', '45 deg', '90 deg', '135 deg'};
ilabelname={'0 deg', '45 deg', '90 deg', '135 deg','mixed'};
ROIsizes=[];
ortho_idxi=cell(11,1);
for k=1:11
ROIsize=size(allRall{k,1}.staticgratings_002.Fcz,1);
ROIsizes=[ROIsizes;ROIsize];
if k==1
ortho_idxi{k,1}=ortho_idx(1:ROIsizes);
else
ortho_idxi{k,1}=ortho_idx(sum(ROIsizes(1:k-1))+1:sum(ROIsizes(1:k)));
end
end

ortho_data = [];
ortho_ROI_all=cell(4,1);
ortho_ROI_oi=cell(5,1);
odata_async_ori_all=cell(4,1);% cell 크기는 pref의 가짓수
odata_sync_ori_all=cell(4,1);
odata_async_ori=cell(5,1);% cell 크기는 sitm의 가짓수
odata_sync_ori=cell(5,1);

for o = 1:4
    for ilabel= 1:5
        ortho_ROI_o=[];
        odata_async_o=[];
        odata_sync_o=[];
        for k=1:11
    ortho_ROI = find(ortho_idxi{k,1} == o);
    ortho_ROI_o=[ortho_ROI_o; ortho_ROI];
    Nvaltrials = size(allFczdata{k,1}.artlines0_Fczcell, 2);
    trialsoi_sync = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(2*ilabel-1);
    trialsoi_async = allExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(2*ilabel);
    odata_async = mean(allFczdata{k,1}.Fczcell(ortho_ROI, trialsoi_async), 2);
    odata_sync = mean(allFczdata{k,1}.Fczcell(ortho_ROI, trialsoi_sync), 2);
    odata_async_o=[odata_async_o;odata_async];
    odata_sync_o=[odata_sync_o;odata_sync];
        end
    ortho_ROI_oi{ilabel,1}=ortho_ROI_o;
    odata_async_ori{ilabel,1}=odata_async_o;
    odata_sync_ori{ilabel,1}=odata_sync_o;
    end
    ortho_ROI_all{o,1}=ortho_ROI_oi;
    odata_async_ori_all{o,1}=odata_async_ori;
    odata_sync_ori_all{o,1}=odata_sync_ori;
end

for o = 1:4
    for ilabel= 1:5
    p_value = ranksum(data_sync_ori_all{o,1}{ilabel,1},odata_sync_ori_all{o,1}{ilabel,1});%Q//그런데 이 방식은 그냥 ROI를 다르게 지정해버리는 방식 같음.favor하는 ori를 다르게.
    disp(sprintf('sync p-value for %s pref vs ortho %s trialtype orientation: %f', oname{o}, ilabelname{ilabel}, p_value));

    a_p_value = ranksum(data_async_ori_all{o,1}{ilabel,1},odata_async_ori_all{o,1}{ilabel,1});
    disp(sprintf('sync p-value for %s pref vs ortho %s trialtype orientation: %f', oname{o}, ilabelname{ilabel}, a_p_value));
    end
end
    % 예전 코드.Q
for o=1:4
    trialori = max_idx(:, 1) == o ; 
    max_data = combined_data(trialori, o);
    ortho_data = combined_data(trialori, mod(o+1, 4) + 1);
    p_value = ranksum(max_data,ortho_data);
    disp(['p-value for pref vs ortho orientation ' oname{o} ': ' num2str(p_value)]);
end
%% plot TargetStimmability -SGholo 다시 분석하기 
Nvaltrials=size(Rholormart.Fczcell,2);
holocondinds = find(ExpStruct.holoStimParams.powers>0);
Nholoconds=length(holocondinds);
% just the targets on targeted trials
figure('Position', [0 0 2000 800])
    annotation('textbox', [0.1 0.91 0.9 0.1], 'string', ...
        sprintf('Stim Activity: N=%d/%d/%d Stimmable/Matched-to-Cells/Targets', ...
        numel(stimmedtargetneuronsrmart.Fczcell), nnz(validtargets), length(validtargets)), ...
        'edgecolor', 'none', 'FontSize', 14)
    % syn랑 asyn SGholo의 경우 psthavgholormart를 사용함. 

for iholo = 1:Nholoconds
    ilabel = ceil(iholo/length(holoRequest.multiplexgroups));
    impg = mod(iholo-1, length(holoRequest.multiplexgroups))+1;
    neuoind = offlineHoloRequest.targetneurons(validtargets & offlineHoloRequest.holoGroups'==ilabel);
    stimmedneuoind = neuoind(Pwsrholormart.Fczcell(neuoind,iholo)<0.05 & Pleftwsrholormart.Fczcell(neuoind,iholo)<0.05);
    if numel(neuoind)==0
        continue
    end
trialsoi = ExpStruct.trialCond(1:Nvaltrials)==holocondinds(iholo);
subplot(length(holoRequest.multiplexgroups), numel(offlineHoloRequest.labels), numel(offlineHoloRequest.labels)*(impg-1)+ilabel)
hold all

plot(psthrmarttimeline, squeeze(mean(psthrmart.Fczcell(neuoind,trialsoi,:), 2)))
plot(psthrmarttimeline, squeeze(mean(psthrmart.Fczcell(neuoind,trialsoi,:), [1,2])), 'k-', 'LineWidth', 1)
xlabel('Time (s)')
ylabel('Fcz')
title(sprintf('%s hologroup\nN=%d/%d/%d stimmable/matched/targets', ...
    offlineHoloRequest.labels{ilabel}, numel(stimmedneuoind), numel(neuoind), nnz(offlineHoloRequest.holoGroups'==ilabel) ))

end
orient('landscape')
%% 
% for i=1:numel(date)
% load 'offlineSGholo.mat'
% 'stimmedtargetneurons', 'stimmedholoconds')
% end
