clear all; close all; %clc
%%
% date={'230111/','230110/','230109/','230108/','230107/'};
% mice='MU31_1/';
date={'230111/','230110/','230109/','230107/','230106/'};
mice='MU31_2/';
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
    
    % psthtimeline_mouse=cell(numel(date), 1);
    Ravgholormart_mouse=cell(numel(date), 1);
    expstruct_mouse = cell(numel(date), 1);
    fczdata_mouse = cell(numel(date), 1);
    psth_mouse = cell(numel(date), 1);
    psthrmarttimeline_mouse = cell(numel(date), 1);
    valid_mouse = cell(numel(date), 1);
    % offlineholoreq_mouse = cell(numel(date), 1);
    vis_mouse= cell(numel(date), 1);
    Rall_mouse=cell(numel(date), 1);

    %%
for i = 1:numel(date)%%%% MU31_1의 230110일 경우 SGholo2로 다 들여야함. 
pathpp = ['//shinlab/ShinLab/MesoHoloExpts/mesoholoexpts_postprocessed/' mice date{i}];

% load([pathpp 'offlineSGholo_rmart.mat'],'Ravgholormart','Rholormart')
% load([pathpp 'offlineSGholo_psth_rmart.mat'])
% load([pathpp 'offlineICholo.mat'], 'validICtargets')
load([pathpp 'postprocessed_psth.mat'],'psthall')
load([pathpp 'postprocessed.mat'],'vis','Rall')
if strcmp(mice, 'MU31_2/') && strcmp(date{i}, '230106/')
load([pathpp 'rmartifact_stimtest_5cph.mat'],'psthrmart')
else
load([pathpp 'rmartifact_stimtest_ICholo_5cph.mat'],'psthrmart')
end

holodaqpath= '//shinlab/ShinLab/MesoHoloExpts/mesoholoexpts_scanimage/TempHoloOutfiles/';
holodaqfolder= date{i};

if strcmp(mice, 'MU31_2/') && strcmp(date{i}, '230111/')
    mousedata = strcat('/', 'stimtest_ICholo_5cph/');
    filename = [strrep(date{i}, '/', ['_','stimtest_ICholo_5cph_A']) '.mat'];
    load([holodaqpath holodaqfolder mousedata filename])
elseif strcmp(mice, 'MU31_1/') && strcmp(date{i}, '230108/')
    mousedata = 'MU31_1_stimtest/';
    filename ='230108_MU31_1_stimtest_A';
    load([holodaqpath holodaqfolder mousedata filename])
elseif strcmp(mice, 'MU31_2/') && strcmp(date{i}, '230106/')
mousedata = 'MU31_2_stimtest_5cph/';
filename ='230106_MU31_2_stimtest_5cph_A';
load([holodaqpath holodaqfolder mousedata filename])
else
mousedata = strrep(mice, '/', '_stimtest_ICholo_5cph/');
filename = [strrep(date{i}, '/', ['_', mice(1:end-1), '_','stimtest_ICholo_5cph_A']) '.mat'];
load([holodaqpath holodaqfolder mousedata filename])
end

%psthrmarttimeline기준을 Fczcell로 했음
% psthtimeline_mouse{i} = psthall.("stimtest_ICholo_5cph_x").psthtimeline;
if strcmp(mice, 'MU31_2/') && strcmp(date{i}, '230106/')
psthrmarttimeline_mouse{i} = (0:size(psthrmart.Fczcell,3)-1)*mean(diff(psthall.("stimtest_5cph_x").psthtimeline));
else
psthrmarttimeline_mouse{i} = (0:size(psthrmart.Fczcell,3)-1)*mean(diff(psthall.("stimtest_ICholo_5cph_x").psthtimeline));%what is ICholo_merged?
end
expstruct_mouse{i} = ExpStruct;
% valid_mouse{i} = validICtargets;
vis_mouse{i}=vis;
Rall_mouse{i}=Rall;
% Ravgholormart_mouse{i}=Ravgholormart;
% fczdata_mouse{i}.Fczcell = Rholormart.Fczcell;
% fczdata_mouse{i}.artlines0_Fczcell = Rholormart.artlines0_Fczcell;

psth_mouse{i}.Fczcell = psthrmart.Fczcell;
psth_mouse{i}.artlines0_Fczcell = psthrmart.artlines0_Fczcell;

% for i = 1:numel(date)
% if strcmp(mice, 'MU31_1/') && strcmp(date{i}, '230110/')
%     SG ='SGholo2_x';
%     % filename = [strrep(date{i}, '/', ['_', mice(1:end-1), '_','SGholo2_A']) '.mat'];
% else
% % SG =SGholoexptidn;
% end
% pathpp = ['//shinlab/ShinLab/MesoHoloExpts/mesoholoexpts_postprocessed/' mice date{i}];
% load(sprintf('%sofflineHoloRequest_%s.mat',pathpp,SG))
% offlineholoreq_mouse{i} = offlineHoloRequest;
end

    %% run unitl here and then go back to load other mouses
    allExpstruct = [allExpstruct; expstruct_mouse];
    allFczdata = [allFczdata; fczdata_mouse];
    allpsth = [allpsth; psth_mouse];
    allpsthrmarttimeline = [allpsthrmarttimeline; psthrmarttimeline_mouse];
    % allvalid = [allvalid; valid_mouse];
    % allofflineholoreq = [allofflineholoreq; offlineholoreq_mouse];
    allRavgholormart=[allRavgholormart;Ravgholormart_mouse];
    allvis=[allvis;vis_mouse];
    allRall=[allRall;Rall_mouse];
    % allpsthtimeline = [allpsthtimeline; psthtimeline_mouse];%just in case. might delete later
    %% load off and online data 
% for each ONLINE ROI, find the OFFLINE ROI that is nearest.
% allneuronXYcoords=[];
% alliscell=[];
% alloffline=[];
% allonlineSGHR=[];
% allontarginds=[];

% date={'230111/','230110/','230109/','230108/','230107/'};
% mice='MU31_1/';
date={'230111/','230110/','230109/','230107/','230106/'};
mice='MU31_2/';

ontarginds_mouse=cell(numel(date),1);
neuronXYcoords_mouse=cell(numel(date), 1);
iscell_mouse=cell(numel(date),1);
offline_mouse=cell(numel(date),1);
onlineSGHR_mouse=cell(numel(date),1);
for i = 1:numel(date)
onlinepath = ['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts_scanimage\' mice date{i} '\ClosedLoop_justgreen\'];
load([onlinepath 'online_params.mat'],'neuronXYcoords')
filename = ['holoRequest_' mice(1:end-1), '_'  date{i}(1:end-1) '_' 'staticICtxi_001' '.mat'];
load([onlinepath filename])
ontarginds_mouse{i}=ontarginds;
% [SGholoreqfile,SGholoreqpath] = uigetfile([onlinepath '*.mat'], 'choose SG holoRequest file from onlinepath');
% onlineSGHR = load([SGholoreqpath SGholoreqfile]);
neuronXYcoords_mouse{i}=neuronXYcoords;
load(['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts_postprocessed\' mice date{i} '\postsuite2p_params.mat'])
iscell_mouse{i}=iscell;
if strcmp(mice, 'MU31_2/') && strcmp(date{i}, '230106/')
offlinepath = ['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts\' mice date{i} '\stimtest_5cph\'];    
else
offlinepath = ['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts\' mice date{i} '\stimtest_ICholo_5cph\'];
end
offline = load([offlinepath 'Fall_splitx.mat']);
offline_mouse{i}=offline;
% onlineSGHR_mouse{i}=onlineSGHR;
end

%%
allneuronXYcoords = [allneuronXYcoords; neuronXYcoords_mouse];
alliscell=[alliscell;iscell_mouse];
alloffline=[alloffline;offline_mouse];
allonlineSGHR=[allonlineSGHR;onlineSGHR_mouse];
% allontarginds=[allontarginds;ontarginds_mouse];
%%
all_Noffrois=cell(10,1);%number of all dates and mouse
all_imoffroi=cell(10,1);
all_offroictr=cell(10,1);
for i =1:10
all_Noffrois{i,1} = numel(alloffline{i,1}.stat);
all_imoffroi{i,1} = zeros(alloffline{i,1}.ops.Ly, alloffline{i,1}.ops.Lx);
all_offroictr{i,1} = zeros(all_Noffrois{i,1},2);
end

for i =1:10
for ci = 1:all_Noffrois{i,1}
tempiminds = sub2ind(size(all_imoffroi{i,1}),alloffline{i,1}.stat{ci}.ypix, alloffline{i,1}.stat{ci}.xpix);
all_imoffroi{i,1}(tempiminds) = ci;
all_offroictr{i,1}(ci,:)=double(alloffline{i,1}.stat{ci}.med);
end
end

cropedgethr = 50;
fname = "\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts_scanimage\MU31_2\230111\stimtest_ICholo_5cph\file_00001.tif";

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
all_targetROIdist=cell(10,1);
alltargetroiXYdist=cell(10,1);
% x/yumperpix가 trial마다 동일하다 가정함
load(['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts_scanimage\MU31_2\230111\ClosedLoop_justgreen\holoRequest_MU31_2_230111_staticICtxi_001.mat'],'fullxsize_orig','fullysize_orig')
xumperpix = fullxsize_orig/size(offline.ops.meanImg,2);
yumperpix = fullysize_orig/size(offline.ops.meanImg,1);
%% need to check if this is okay
% date={'230111/','230110/','230109/','230108/','230107/'};
% mice='MU31_1/';
date={'230111/','230110/','230109/','230107/','230106/'};
mice='MU31_2/';
% allxynew=[];
xynew_mouse=cell(numel(date),1);
% allofflineholoreq{i,1}=allExpstruct{i,1}.holoRequestOrig;
for i = 1:numel(date)
onlinepath = ['\\shinlab\ShinLab\MesoHoloExpts\mesoholoexpts_scanimage\' mice date{i} '\ClosedLoop_justgreen\'];
filename = ['holoRequest_' mice(1:end-1), '_'  date{i}(1:end-1) '_' 'staticICtxi_001' '.mat'];
load([onlinepath filename])
xynew_mouse{i}=xynew;
end
allxynew=[allxynew;xynew_mouse];
%%
for i =1:10
% target과 ROI간의 거리
% allofflineholoreq{i,1}.targets= fliplr(allxynew{i,1});
% all_targetROIdist{i,1} = sqrt((yumperpix*(allofflineholoreq{i,1}.targets(:,1)-all_offXYcoords{i,1}(:,1)')).^2 + ...
    % (xumperpix*(allofflineholoreq{i,1}.targets(:,2)-all_offXYcoords{i,1}(:,2)')).^2);
all_targetROIdist{i,1} = sqrt((yumperpix*(allExpstruct{i,1}.holoRequest.targets(:,1)-all_offXYcoords{i,1}(:,1)')).^2 + ...
(xumperpix*(allExpstruct{i,1}.holoRequest.targets(:,2)-all_offXYcoords{i,1}(:,2)')).^2);
alltargetroiXYdist{i,1} = all_targetROIdist{i,1}(:, alliscell{i,1});
end %stimtest을 기준으로
% onoffroiumdist = 2*sqrt( (yumperpix*(neuronXYcoords(:,1)-offXYcoords(:,1)')).^2 + ...
%     (xumperpix*(neuronXYcoords(:,2)-offXYcoords(:,2)')).^2);%원래 코드
% ICtargetroiXYdist = onoffroiumdist(onlineICHR.ontarginds, iscell);
% 
% [target2celldist, target2cell]=min(ICtargetroiXYdist, [],2);
% offlineICHoloRequest.target2celldist = target2celldist;
% offlineICHoloRequest.targetneurons = target2cell;
% neuroninds = find(iscell);
% offlineICHoloRequest.targetROIs = neuroninds(target2cell);
%%
allvalid=cell(10,1);
for i=1:10
onoffroiumdist = 2*sqrt( (yumperpix*(allneuronXYcoords{i,1}(:,1)-all_offXYcoords{i,1}(:,1)')).^2 + ...
    (xumperpix*(allneuronXYcoords{i,1}(:,2)-all_offXYcoords{i,1}(:,2)')).^2);%이거 2곱한 상태인거 같은데 나중에 20으로 자르는 걸로 봐서 10이되는듯
targetroiXYdist{i,1} = onoffroiumdist(allontarginds{i,1}, alliscell{i,1});%ontarginds??
[target2celldist, target2cell]=min(targetroiXYdist{i,1}, [],2);
allvalid{i,1} = target2celldist<20;
alltargetneurons{i,1}=target2cell;
end % ICholo기준으로 이었는데 ontargind를 바꿈
%아니면 굳이 구하지 말고 ICholo의 validtarget를 가져오는 건 어떤가

%% psth heatmap
%find holo conditions (orientation)
save_path='d:\Users\USER\Documents\MATLAB\stimtest';
psth_s=[];
psth_a_accumulated = [];
psth_s_accumulated = [];
Nholoconds = cell(10, 1);
for k = 1:10 
holocondinds = find(allExpstruct{k,1}.holoStimParams.powers >= 0);
Nholoconds{k} = length(holocondinds);
end

for k =1:10
for iholo=1:Nholoconds{k} 
    psth_s=[];
 %추후에 수정 필요
       neuoind = alltargetneurons{k,1};
       [trialsoi,t] = find(allExpstruct{k,1}.trialCond(:) == iholo);
       psth = squeeze(mean(allpsth{k,1}.artlines0_Fczcell(neuoind, trialsoi,:), 2));%for dates with invalid trials I did 1+trialsoi
       psth_s=horzcat(psth_s,psth);
% end
psth_s_accumulated{iholo,1} = psth_s;
end

for p=1:4
subplot(1, 4, p);
imagesc(psth_s_accumulated{p,1});
title(sprintf('PSTH %s stimtest', num2str(allExpstruct{k, 1}.holoStimParams.powers(p))));
end
filename = sprintf('MU31_2_stimtest_ICholo_5cph %s psth.png',date{k-5}(1:end-1));
% saveas(gcf, fullfile(save_path, filename));
end


%% activity for stimtest
poweract_alldates=cell(10,1);
for k=1:10
    if k == 5 || k == 8 || k== 10
        continue;
    end
poweract=cell(Nholoconds{k}-1,1);%power가 0인 경우를 제외
for p=2:Nholoconds{k} %power가 0인 경우를 제외
labeleddist=alltargetroiXYdist{k,1}(:,:); 
% neuoind = alltargetneurons{k,1}(allvalid{k,1});%문제. 지금 alltargetroiXYdist랑 alltargetneurons랑 allvalid가 안 맞는 날들이 있음. 5번째랑 8번째.
Nvaltrials = size(allpsth{k,1}.artlines0_Fczcell, 2);
[trialsoi,t] = find(allExpstruct{k,1}.trialCond(:) == p);
% neuoind = alltargetneurons{k,1}; %all valid를 해야 할까요...안 하니까 시간대별로 전체 수가
% 깨끗하게 맞는 거 같음->valid 하지 말라고 하심
neuoind = allICofflineholoreq{k,1}.targetneurons;
first_t = allExpstruct{k, 1}.outParams.firstStimTimes{1, p}(1, 1);
inter = allExpstruct{k, 1}.outParams.ipi(p) / 100;

indices = [];
for i = 1:numel(allExpstruct{k, 1}.outParams.firstStimTimes{1, p})
    % 현재 first_t에 대한 인덱스 찾기
    indice = min(find(allpsthrmarttimeline{k, 1} >= first_t));
    indices = [indices; indice];
    first_t = first_t + inter;
end

alltargetactivity_alltime=cell(floor(numel(neuoind)/5),1);%이거는 모든 타겟들에 대해 5셀씩 묶어서 stimulation한 시간순으로 활동을 나눈 것.
allmindist=cell(floor(numel(neuoind)/5),1);
for cell5=1:5:numel(neuoind)
    targetactivity_alltime=[];
    if mod(numel(neuoind),5) ~=0 && cell5+5>numel(neuoind)
        cellend=numel(neuoind);
    else
        cellend=cell5+4;
    end
if k == 1 || k == 5 % for cases that have invalid trials  
Trialsoi=trialsoi+1;
else
Trialsoi=trialsoi;
end
    for idc=1:numel(indices)
        % if idc==numel(indices)
        % activity= mean(squeeze(mean(allpsth{k,1}.artlines0_Fczcell(neuoind(cell5:cellend,1),Trialsoi,indices(idc):indices(idc)+2),2)),2);%Fczdata와 동일한 time period
        % 임의로 마지막 indices에서는 indices~indices+2까지를 범위로 함.
        % else
        % 여기가 문제임    
        % activity= mean(squeeze(mean(allpsth{k,1}.artlines0_Fczcell(neuoind(cell5:cellend,1),Trialsoi,indices(idc):indices(idc+1)-1),2)),2);%Fczdata와 동일한 time period
    % if cell5==cellend && mod(cellend,5)==1
    %     activity= mean(squeeze(mean(allpsth{k,1}.artlines0_Fczcell(neuoind(cell5:cellend,1),Trialsoi,indices(idc):indices(idc)+2),2)),1);
    % else
        activity= mean(allpsth{k,1}.artlines0_Fczcell(neuoind(cell5:cellend,1),Trialsoi,indices(idc):indices(idc)+2),3);%얘만 주석 풀고 진행
    % end
        % end
    % targetactivity_alltime=horzcat(targetactivity_alltime,activity);
    if idc== floor(cell5/5)+1
        alltargetactivity_alltime{floor(cell5/5)+1}=activity;
    else
        continue;
    end
    end
    % alltargetactivity_alltime{floor(cell5/5)+1}=targetactivity_alltime(:,floor(cell5/5)+1);
    alltargetactivity=vertcat(alltargetactivity_alltime{:});
    target5dist=labeleddist(cell5:cellend,:);
    mindist=  min(target5dist,[],1);
allmindist{floor(cell5/5)+1}=mindist;
end
poweract{p}=alltargetactivity;
end
poweract_alldates{k}=poweract;
end
%% ICholo data loading for comparison
% date={'230111/','230110/','230109/','230108/','230107/'};
% mice='MU31_1/';
date={'230111/','230110/','230109/','230107/','230106/'};
mice='MU31_2/';

% allICExpstruct={};
% allICofflineholoreq={};
% allICFczdata={};
% allICpsth={};
%allICpsthrmarttimline={};
expstruct_mouse = cell(numel(date), 1);
offlineholoreq_mouse= cell(numel(date), 1);
fczdata_mouse = cell(numel(date), 1);
ICpsth_mouse=cell(numel(date),1);
 ICpsthrmarttimline_mouse=cell(numel(date),1);

for i = 1:numel(date)%%%% MU31_1의 230110일 경우 SGholo2로 다 들여야함. 
pathpp = ['//shinlab/ShinLab/MesoHoloExpts/mesoholoexpts_postprocessed/' mice date{i}];
holodaqpath= '//shinlab/ShinLab/MesoHoloExpts/mesoholoexpts_scanimage/TempHoloOutfiles/';
holodaqfolder= date{i};

if strcmp(mice, 'MU31_1/') && strcmp(date{i}, '230110/')
    IC ='ICholo_merged';
else
IC ='ICholo_x';
end
load(sprintf('%sofflineHoloRequest_%s.mat',pathpp,IC))
offlineholoreq_mouse{i} = offlineICHoloRequest;

load([pathpp 'offlineICholo_rmart.mat'],'RICholormart')
fczdata_mouse{i}.Fczcell = RICholormart.Fczcell;
fczdata_mouse{i}.artlines0_Fczcell = RICholormart.artlines0_Fczcell;

mousedata = strrep(mice, '/', '_ICholo/');
filename = [strrep(date{i}, '/', ['_', mice(1:end-1), '_','ICholo_A']) '.mat'];
load([holodaqpath holodaqfolder mousedata filename])
expstruct_mouse{i} = ExpStruct;

load([pathpp 'rmartifact_ICholo.mat'],'psthrmart')
ICpsth_mouse{i}.Fczcell=psthrmart.Fczcell;
ICpsth_mouse{i}.artlines0_Fczcell=psthrmart.artlines0_Fczcell;
    load([pathpp 'postprocessed_psth.mat'],'psthall')
    ICpsthrmarttimline_mouse{i} = (0:size(psthrmart.Fczcell,3)-1)*mean(diff(psthall.("ICholo_x").psthtimeline));

    if strcmp(mice, 'MU31_1/') && strcmp(date{i}, '230110/')
    filenameB = [strrep(date{i}, '/', ['_', mice(1:end-1), '_','ICholo_B']) '.mat'];
    load([holodaqpath holodaqfolder mousedata filenameB])
    B_expstruct = ExpStruct;

    load([pathpp 'rmartifact_ICholo_merged.mat'],'psthrmart')
    ICpsth_mouse{i}.Fczcell=psthrmart.Fczcell;
    ICpsth_mouse{i}.artlines0_Fczcell=psthrmart.artlines0_Fczcell;

    else 

    continue;
    end
end

allICExpstruct = [allICExpstruct; expstruct_mouse];
allICofflineholoreq=[allICofflineholoreq;offlineholoreq_mouse];
allICFczdata=[allICFczdata;fczdata_mouse];
allICpsth=[allICpsth;ICpsth_mouse];
allICpsthrmarttimline=[allICpsthrmarttimline;ICpsthrmarttimline_mouse];

%% ICholo activity 구하는 부분
all_sdata = cell(10,1);
all_asdata = cell(10,1);
Nholocodinds = cell(10, 1);
Nholoconds = cell(10, 1);
for k = 1:10
holocondinds = find(allICExpstruct{k,1}.holoStimParams.powers > 0);
Nholocodinds{k} = holocondinds;
Nholoconds{k}=length(holocondinds);
end%날짜 상관없이 같음 확인.

%모든 cell에 대해 AUROC를 먼저 구하고, 해당되는 뉴런들만 추출
for k=1:10
    allsholos_data=[];
    allasholos_data=[];
for  iholo = 1:2:Nholoconds{k}%need to fix
        Nvaltrials = size(allICFczdata{k,1}.artlines0_Fczcell, 2);
        if k==2 %(마우스가 1번이고 9일에 해당되는 경우)
        ABExpstruct=horzcat(allICExpstruct{k,1}.trialCond,B_expstruct.trialCond);
        trialsoi_sync = ABExpstruct(1:Nvaltrials) == holocondinds(iholo);%ICholo도 마찬가지로 홀수가 싱크인지 확인하기
        trialsoi_async = ABExpstruct(1:Nvaltrials) == holocondinds(iholo+1);
        else
        trialsoi_sync = allICExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo);%ICholo도 마찬가지로 홀수가 싱크인지 확인하기
        trialsoi_async = allICExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo+1);
        end

        ilabel = ceil(iholo/length(allICExpstruct{k,1}.holoRequest.multiplexgroups));
        % neuoind = allICofflineholoreq{k,1}.targetneurons(allICvalid{k,1}
        % & allICofflineholoreq{k,1}.holoGroups'==ilabel);% 이것도 확인...
        neuoind = allICofflineholoreq{k,1}.targetneurons(allICofflineholoreq{k,1}.holoGroups'==ilabel);

        data_async = mean(allICFczdata{k,1}.artlines0_Fczcell(neuoind, trialsoi_async,:), 2);%여기서 power로 잘라야 함.
        data_sync = mean(allICFczdata{k,1}.artlines0_Fczcell(neuoind, trialsoi_sync,:), 2); 
        allsholos_data=[allsholos_data;data_sync];
        allasholos_data=[allasholos_data;data_async];

end     
all_sdata{k,1} =allsholos_data; 
all_asdata{k,1} =allasholos_data; 
end

%% AUROC values between stimtest and ICholo
allAUROCiholo=cell(10,1);
allPmwwICholo=cell(10,1);
Nholoconds = cell(10, 1);
date_allneuoind=cell(10, 1);
for k = 1:10
holocondinds = find(allICExpstruct{k,1}.holoStimParams.powers > 0);
Nholocodinds{k} = holocondinds;
Nholoconds{k}=length(holocondinds);
end%날짜 상관없이 같음 확인.

for k=1:10
    if k == 5 || k == 8 || k== 10
        continue;
    end
Nneurons=size(all_sdata{k,1},1);%
AUROCICholo = NaN(Nneurons,Nholoconds{k}/2);
PmwwICholo = NaN(Nneurons,Nholoconds{k}/2);
m=find(allExpstruct{k, 1}.holoStimParams.powers(:)==allICExpstruct{k, 1}.holoStimParams.powers(1,2));%power 일치

allneuoind=[];
for iholo = 1:2:Nholoconds{k}
ilabel = ceil(iholo/length(allICExpstruct{k,1}.holoRequest.multiplexgroups));
neuoind = allICofflineholoreq{k,1}.targetneurons(allICofflineholoreq{k,1}.holoGroups'==ilabel);
allneuoind=[allneuoind;neuoind];
end
date_allneuoind{k,1}=allneuoind;

for iholo = 1:2:Nholoconds{k}
Nvaltrials = size(allICFczdata{k,1}.artlines0_Fczcell, 2);
ilabel = ceil(iholo/length(allICExpstruct{k,1}.holoRequest.multiplexgroups));
if k==2 %(마우스가 1번이고 9일에 해당되는 경우)
ABExpstruct=horzcat(allICExpstruct{k,1}.trialCond,B_expstruct.trialCond);
trialsoi_sync = ABExpstruct(1:Nvaltrials) == holocondinds(iholo);%ICholo도 마찬가지로 홀수가 싱크인지 확인하기
trialsoi_async = ABExpstruct(1:Nvaltrials) == holocondinds(iholo+1);
else
trialsoi_sync = allICExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo);%ICholo도 마찬가지로 홀수가 싱크인지 확인하기
trialsoi_async = allICExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo+1);
end
trialsoi_stim=1:3;
for ci = 1:Nneurons
p = ranksum(allICFczdata{k,1}.artlines0_Fczcell(date_allneuoind{k,1}(ci),trialsoi_sync), poweract_alldates{k,1}{m,1}(ci,trialsoi_stim));
PmwwICholo(ci,ilabel) = p;
[X,Y,T,AUC] = perfcurve([ones(1,nnz(trialsoi_sync)) zeros(1,nnz(trialsoi_stim))], ...
[allICFczdata{k,1}.artlines0_Fczcell(date_allneuoind{k,1}(ci),trialsoi_sync) poweract_alldates{k,1}{m,1}(ci,trialsoi_stim)], '1');
AUROCICholo(ci,ilabel) = AUC;
end
allAUROCiholo{k,1}=AUROCICholo;
allPmwwICholo{k,1}=PmwwICholo;
end
end
 %%   
name = {'1', '2', '3', '4','5','6','7','8'};
% save_path='d:\Users\USER\Documents\MATLAB\AUROC_SGholo';
for ilabel=1:8
    auroc_values=[];
     p_values=[];
    for k=1:10
        if k == 5 || k == 8 || k== 10
        continue;
        end
        % neuoind = allICofflineholoreq{k,1}.targetneurons(allICofflineholoreq{k,1}.holoGroups'==ilabel);
        auroc_value=allAUROCiholo{k,1}(:,ilabel);
        auroc_values=[auroc_values;auroc_value];
        p_value=allPmwwICholo{k,1}(:,ilabel);
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
    subplot(4, 2, ilabel);
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
%% AUROC values between ICholo
allAUROCiholo=cell(10,1);
allPmwwICholo=cell(10,1);
Nholoconds = cell(10, 1);

for k = 1:10
holocondinds = find(allICExpstruct{k,1}.holoStimParams.powers > 0);
Nholocodinds{k} = holocondinds;
Nholoconds{k}=length(holocondinds);
end%날짜 상관없이 같음 확인.

for k=1:10
Nneurons=size(allICFczdata{k,1}.artlines0_Fczcell,1);
AUROCICholo = NaN(Nneurons,Nholoconds{k}/2);
PmwwICholo = NaN(Nneurons,Nholoconds{k}/2);

for iholo = 1:2:Nholoconds{k}
Nvaltrials = size(allICFczdata{k,1}.artlines0_Fczcell, 2);
ilabel = ceil(iholo/length(allICExpstruct{k,1}.holoRequest.multiplexgroups));
if k==2 %(마우스가 1번이고 9일에 해당되는 경우)
ABExpstruct=horzcat(allICExpstruct{k,1}.trialCond,B_expstruct.trialCond);
trialsoi_sync = ABExpstruct(1:Nvaltrials) == holocondinds(iholo);%ICholo도 마찬가지로 홀수가 싱크인지 확인하기
trialsoi_async = ABExpstruct(1:Nvaltrials) == holocondinds(iholo+1);
else
trialsoi_sync = allICExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo);%ICholo도 마찬가지로 홀수가 싱크인지 확인하기
trialsoi_async = allICExpstruct{k,1}.trialCond(1:Nvaltrials) == holocondinds(iholo+1);
end

for ci = 1:Nneurons
p = ranksum(allICFczdata{k,1}.artlines0_Fczcell(ci,trialsoi_sync),allICFczdata{k,1}.artlines0_Fczcell(ci,trialsoi_async));
PmwwICholo(ci,ilabel) = p;
[X,Y,T,AUC] = perfcurve([ones(1,nnz(trialsoi_async)) zeros(1,nnz(trialsoi_sync))], ...
[allICFczdata{k,1}.artlines0_Fczcell(ci,trialsoi_async) allICFczdata{k,1}.artlines0_Fczcell(ci,trialsoi_sync)], '1');
AUROCICholo(ci,ilabel) = AUC;
end
allAUROCiholo{k,1}=AUROCICholo;
allPmwwICholo{k,1}=PmwwICholo;
end
end

 %%   
name = {'indin13','indin14','indin23','indin24','ICencoder1','RCencoder1','RCencoder2','ICencoder2'};
% save_path='d:\Users\USER\Documents\MATLAB\AUROC_SGholo';
for ilabel=1:8
    auroc_values=[];
     p_values=[];
         for k=1:10
        neuoind = allICofflineholoreq{k,1}.targetneurons(allICvalid{k,1}& allICofflineholoreq{k,1}.holoGroups'==ilabel);
        auroc_value=allAUROCiholo{k,1}(neuoind,ilabel);
        auroc_values=[auroc_values;auroc_value];
        p_value=allPmwwICholo{k,1}(neuoind,ilabel);
        p_values=[p_values;p_value];
         end
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
    subplot(4, 2, ilabel);
    bar(binEdges(1:end-1), neuron_counts, 'b');
    hold on;
    bar(binEdges(1:end-1), signeuron_counts, 'r');
    hold off;

    xlabel('AUROC');
    ylabel('Neuron Count');
    title(sprintf('%s AUROC Neuron Count',name{ilabel}));

end
filename = sprintf('artlines0_Fczcell prof 2 method with significant cells.png');