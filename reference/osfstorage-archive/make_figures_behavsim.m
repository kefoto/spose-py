% This script runs all relevant analyses to reproduce all figures and major
% analyses in Hebart, Zheng, Pereira, and Baker (2020)

disp('There was a bug in the original code for producing the figures which leads to minor changes in the results of the similarity matrix.')
disp('The bug was that variable ctmp (also in function embedding2sim.m) was not reinitialized in the for loop. The largest change is an r of 0.0011. If any, the ensuing results are better.')
disp('This does not affect any of the reported results but may lead to barely noticeable differences in the tsne plot.')
pause(1)

% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'data');
variable_dir = fullfile(base_dir,'variables');

%% Add relevant toolboxes

% t-SNE from: https://lvdmaaten.github.io/tsne/#implementations
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data

% load embedding
spose_embedding49 = load(fullfile(data_dir,'spose_embedding_49d_sorted.txt'));
% get dot product (i.e. proximity)
dot_product49 = spose_embedding49*spose_embedding49';
% load similarity computed from embedding (using embedding2sim.m)
load(fullfile(data_dir,'spose_similarity.mat'))
dissim = 1-spose_sim;
% load test set
triplet_testdata49 = load(fullfile(data_dir,'data1854_batch5_test10.txt'))+1; % 0 index -> 1 index
% load 48 object set, also get their indices
load(fullfile(data_dir,'RDM48_triplet.mat'))
load(fullfile(data_dir,'RDM48_triplet_splithalf.mat'))
% load typicality ratings for 27 categories
load(fullfile(data_dir,'typicality_data27.mat'))
% load ratings of 20 subjects for 20 objects along 49 dimensions
load(fullfile(data_dir,'dimension_ratings.mat'))
% load answers from participants labeling the images
load(fullfile(data_dir,'dimlabel_answers.mat'))

%% in the training and test datasets, the order is still wrong, let's change it
load(fullfile(variable_dir,'sortind.mat'));
for i_obj = 1:1854
    triplet_testdata49(triplet_testdata49==sortind(i_obj)) = 10000+i_obj;
end
triplet_testdata49 = triplet_testdata49-10000;

%% get dimension labels, short labels and colors

load(fullfile(variable_dir,'labels.mat'))
load(fullfile(variable_dir,'labels_short.mat'))

h = fopen(fullfile(variable_dir,'colors.txt')); % get list of colors in hexadecimal format

col = zeros(0,3);
while 1
    l = fgetl(h);
    if l == -1, break, end
    
    col(end+1,:) = reshape(sscanf(l(2:end).','%2x'),3,[]).'/255; % hex2rgb
    
end
fclose(h);

col(1,:) = [];
col([1 2 3],:) = col([2 3 1],:);

% now adapt colors
colors = col([1 20 3 38 9 7 62 57 13 6 24 25 50 48 36 53 46 28 62 18 15 58 2 11 40 45 27 55 36 30 34 31 41 16 27 61 17 36 57 25 63],:); colors(end+1:49,:) = col([8:56-length(colors)],:);
colors(46,:) = colors(46,:)-0.2; % medicine related is too bright, needs to be darker

clear col h l


%% Load smaller version of images, words, and unique IDs for each image

load(fullfile(variable_dir,'im.mat'))
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'unique_id.mat'))
load(fullfile(variable_dir,'words48.mat'))

% TODO: perhaps still sorting error with im and imwords
% now sort images according to unique_id
[~,i] = sort(unique_id);
[~,j] = sort(imwords);
imwords(i) = imwords(j);
im(i) = im(j);

[~,~,wordposition48] = intersect(words48,words,'stable');



%% Figure 1: Get embedding and relevant vectors

dosave = 0;

if dosave
    hf = figure;
    hf.Position(3:4) = [300 200];
    hf.Color = 'none';
    % plot only part to actually see width
    rng(1)
    rn = randperm(1854); rn = sort(rn(1:100));
    imagesc(spose_embedding49(rn,3:3:end-5),[0 0.5])
    ha = gca;
    colormap(viridis)
    axis equal off
    saveas(hf,'Figure1_panelb1.svg') %#ok<*UNRCH>
    close(hf)
end

% Get random embedding
if dosave
    hf = figure;
    hf.Position(3:4) = [300 200];
    hf.Color = 'none';
    % plot only part to actually see width
    rng(1)
    tmp = rand(100,30);
    imagesc(tmp,[0.5 1])
    ha = gca;
    colormap(viridis)
    axis equal off
    hf.Position(3:4) = [900 1200];
    saveas(hf,'Figure1_panelc1.svg')
    % get three rows
    imagesc(tmp(47,:),[0.5 1])
    axis equal
    saveas(hf,'Figure1_panelc2.svg')
    imagesc(tmp(60,:),[0.5 1])
    axis equal
    saveas(hf,'Figure1_panelc3.svg')
    imagesc(tmp(86,:),[0.5 1])
    axis equal
    saveas(hf,'Figure1_panelc4.svg')
    close(hf)
end

% Get similarity plot

ind = clustering_algorithm(3,5,spose_embedding49); % somewhat arbitrary way of sorting objects, according to 3 most dominant dimensions in each object
hf = figure; hf.Position(3:4) = [600 600]; imagesc(spose_sim(ind(1:10:end),ind(1:10:end)),[0 0.9])
colormap(viridis)
axis equal off
if dosave
    saveas(hf,'Figure1_panelb2.svg')
end

% Get plot of images for preview
IM = zeros(30*150,61*150,3);
ipos = -149;
jpos = 1;
cnt = 0;
for i = 1:30
    for j = 1:61
        cnt = cnt+1;
        ipos = ipos+150;
        if ipos>size(IM,1)
            ipos = 1;
            jpos = jpos+150;
        end
        IM(ipos:ipos+149,jpos:jpos+149,:) = im2double(im{cnt});
    end
end

hf = figure;
imagesc(IM)
axis equal off
if dosave
    saveas(hf,'objects.svg')
end

% Make softmax plot
if dosave
    x = -8:0.1:8;
    y = 1./(1+2*exp(-x));
    hf = figure; plot(x,y,'Color',[0.5 0.5 0.5],'LineWidth',5), axis off
    saveas(gcf,'softmax.svg')
    close(hf)
end

%% Figure 2: Predict behavior and similarity

dosave = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata49),1);
behav_predict_prob = zeros(length(triplet_testdata49),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata49)
    sim(1) = dot_product49(triplet_testdata49(i,1),triplet_testdata49(i,2));
    sim(2) = dot_product49(triplet_testdata49(i,1),triplet_testdata49(i,3));
    sim(3) = dot_product49(triplet_testdata49(i,2),triplet_testdata49(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc = 100*mean(behav_predict==1);
% get prediction for each object
for i_obj = 1:1854
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata49==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata49==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95 = 1.96*std(behav_predict_obj)/sqrt(1854);


%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%

h = fopen(fullfile(data_dir,'triplets_noiseceiling.csv'),'r');
NCdat = zeros(20000,5);
cnt = 0;
while 1
    l = fgetl(h);
    if l == -1, break, end
    l2 = strsplit(l);
    cnt = cnt+1;
    NCdat(cnt,:) = str2double(l2);
end
fclose(h);

% sort each triplet and change choice id
for i = 1:length(NCdat)
    [sorted,sortind] = sort(NCdat(i,1:3));
    NCdat(i,1:4) = [sorted find(sortind==NCdat(i,4))];
end

% get unique ID for each triplet by merging numbers
NCstr = num2cell(num2str(NCdat(:,1:3)),2);
uid = unique(NCstr);

% get number of triplets for each
for i = 1:1000
   nNC(i) = sum(strcmp(NCstr,uid{i}));  
end

% Now run for all just to see what happens (get how many people respond the same)
for i = 1:1000
    ind = strcmp(NCstr,uid{i});
    answers = NCdat(ind,4);
    h = hist(answers,1:3);
    consistency(i,1) = max(h)/sum(h); % the best one divided by all
end

noise_ceiling = mean(consistency)*100;
noise_ceiling_ci95 = 1.96 * std(consistency)*100 / sqrt(1000);

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%
hf = figure;
hf.Position(3:4) = [900 1200];
% first plot noise ceiling
wid = 8;
x = 1+ [-wid wid wid -wid];
nc1 = noise_ceiling+noise_ceiling_ci95;
nc2 = noise_ceiling-noise_ceiling_ci95;
y = [nc1 nc1 nc2 nc2];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hold on
% now plot results
% ha1 = plot(1,behav_predict_train_acc,'o','MarkerFaceColor',[1 0 0],'MarkerEdgeColor','none','MarkerSize',12);
% ha2 = plot(2,behav_predict_acc,'o','MarkerFaceColor',[0 0 0],'MarkerEdgeColor','none','MarkerSize',12);
ha3 = bar(1,behav_predict_acc,'FaceColor',[0 0 0],'EdgeColor','none','BarWidth',6);
hb = errorbar(6,behav_predict_acc,behav_predict_acc_ci95,'Color',[0 0 0],'LineWidth',3);
hb = plot(x(1:2),[33.3333 33.3333],'r','LineWidth',3);
axis equal
xlim(x(1:2))
ylim([30 75])

hax = gca;
hax.TickDir = 'both';
hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1;
hax.Box = 'off';
if dosave
    saveas(hf,'Figure2a.svg')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now compare similarity from model to similarity in behavior %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for that focus on those 48 objects only rather than the entire matrix
tic
esim = exp(dot_product49);
cp = zeros(1854,1854);
ctmp = zeros(1,1854);
for i = 1:1854
    for j = i+1:1854
        ctmp = zeros(1,1854);
        for k_ind = 1:length(wordposition48)
            k = wordposition48(k_ind);
            if k == i || k == j, continue, end
            ctmp(k) = esim(i,j) / ( esim(i,j) + esim(i,k) + esim(j,k) );
        end
        cp(i,j) = sum(ctmp); % run sum first, divide all by 48 later
    end
end
toc
cp = cp/48; % complete the mean
cp = cp+cp'; % symmetric
cp(logical(eye(size(cp)))) = 1;

spose_sim48 = cp(wordposition48,wordposition48);

% compare to "true" similarity
r48 = corr(squareformq(spose_sim48),squareformq(1-RDM48_triplet))

% run 1000 bootstrap samples for confidence intervals, bootstrap cannot 
% be done across objects because otherwise it's biased
rng(2)
rnd = randi(nchoosek(48,2),nchoosek(48,2),1000);
c1 = squareformq(spose_sim48);
c2 = squareformq(1-RDM48_triplet);
for i = 1:1000
    r48_boot(:,i) = corr(c1(rnd(:,i)),c2(rnd(:,i)));
end
r48_ci95_lower = tanh(atanh(r48) - 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI
r48_ci95_upper = tanh(atanh(r48) + 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI


h = figure;
h.Position(3:4) = [1200 768];
ha = subtightplot(1,3,1);
imagesc(spose_sim48,[0 1])
colormap(viridis)
hold on
text(24,-4,'predicted similarity matrix','HorizontalAlignment','center','FontSize',16);
axis off square tight
hb = subtightplot(1,3,2);
imagesc(1-RDM48_triplet,[0 1])
text(24,-4,'measured similarity matrix','HorizontalAlignment','center','FontSize',16);
colormap(viridis)
axis off square tight
hc = subtightplot(1,3,3);
hc.Position(1) = hc.Position(1)+0.02;
% plot(squareformq(spose_sim48),squareformq(1-RDM48_triplet),'o','MarkerFaceColor',[0.1651 0.4674 0.5581],'MarkerEdgeColor','none','MarkerSize',3);
plot(squareformq(spose_sim48),squareformq(1-RDM48_triplet),'o','MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor','none','MarkerSize',3);
hold on
plot([0 1],[0 1],'k')
axis square tight
xlabel('predicted similarity')
ylabel('measured similarity')
legend('R = 0.90','Location','NorthWest')
if dosave
    saveas(h,'Figure2b.svg')
end

% get reliability of each split
reliability48 = corr(squareformq(1-RDM48_triplet_split1),squareformq(1-RDM48_triplet_split2));
splithalf48 = tanh(mean(atanh([corr(squareformq(1-RDM48_triplet_split1),squareformq(spose_sim48)) corr(squareformq(1-RDM48_triplet_split2),squareformq(spose_sim48))])));

% get amount of variance explained (in the correlation or in behavior?)
variance_explained48 = splithalf48.^2 / reliability48.^2;

% get stats
fprintf('Accuracy on test data: %2.2f (95%% CI across objects: %2.2f)\n',mean(behav_predict_obj),behav_predict_acc_ci95) 
fprintf('Noise ceiling: %2.2f (95%% CI across objects: %2.2f)\n',noise_ceiling,noise_ceiling_ci95)


% get 10000 bootstrap samples of behav_predict_obj and consistency
rng(42)
ind_rnd_behav = randi(1854,1000,10000);
ind_rnd_ceiling = randi(1000,1000,10000);
btstp_se = 100*std((mean(behav_predict_obj(ind_rnd_behav))-100/3)./(100*mean(consistency(ind_rnd_ceiling))-100/3));
clear ind_rnd_behav ind_rnd_ceiling


fprintf('Percent performance achieved (subtracting chance): %2.2f (95%% CI across objects: %2.2f)\n',100*mean((mean(behav_predict_obj)-100/3)./(100*mean(consistency)-100/3)),btstp_se)

%% Figure 6: Calculate how the behavioral prediction changes when eliminating dimensions with small weight

dosave = 0;

% do this for each obj separately, i.e. get index for each object across 49
% dimensions, then eliminate one by one

fn1 = fullfile(data_dir,'spose_similarity_reduced.mat');
fn2 = fullfile(data_dir,'spose_embedding49_reduced.mat');

if ~exist(fn1,'file') || ~exist(fn2','file')
    [~,embedding_sortind] = sort(spose_embedding49,2);
    disp('Getting reduced versions of embeddings and converting them to similarity.')
    disp('This takes about 10-15min on a regular laptop but only needs to be run once.')
    for i_dim = 1:49
        fprintf('.')
        % make 49 reduced versions, make 49 reduced similarity matrices
        if i_dim == 1
            spose_embedding49_reduc{i_dim,1} = spose_embedding49;
        else
            spose_embedding49_reduc{i_dim,1} = spose_embedding49_reduc{i_dim-1,1};
        end
        for i = 1:1854
            spose_embedding49_reduc{i_dim,1}(i,embedding_sortind(i,i_dim)) = 0;
        end
        spose_sim_reduc{i_dim,1} = embedding2sim(spose_embedding49_reduc{i_dim,1});
        
    end
    fprintf('\n')
    save(fn1,'spose_sim_reduc')
    save(fn2,'spose_embedding49_reduc')
else
    load(fn1)
    load(fn2)
end

clear sim
for i_dim = 1:49
    rng(42) % for reproducibility
    behav_predict = zeros(length(triplet_testdata49),1);
    dot_product49_reduc = spose_embedding49_reduc{i_dim}*spose_embedding49_reduc{i_dim}';
    for i = 1:length(triplet_testdata49)
        sim(1) = dot_product49_reduc(triplet_testdata49(i,1),triplet_testdata49(i,2));
        sim(2) = dot_product49_reduc(triplet_testdata49(i,1),triplet_testdata49(i,3));
        sim(3) = dot_product49_reduc(triplet_testdata49(i,2),triplet_testdata49(i,3));
        [m,mi] = max(sim);
        if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
        behav_predict(i,1) = mi;
    end
    % get overall prediction (predict choice == 1)
    behav_predict_acc_reduc(i_dim) = 100*mean(behav_predict==1);
    % get prediction for each object
    for i_obj = 1:1854
        behav_predict_obj_reduc(i_obj,i_dim) = 100*mean(behav_predict(any(triplet_testdata49==i_obj,2))==1);
    end
    % get standard error for this value across objects
    behav_predict_acc_reduc_ci95(i_dim,1) = 1.96* std(behav_predict_obj_reduc(:,i_dim))/sqrt(1854);
end

% now reverse it all
behav_predict_acc_reduc = behav_predict_acc_reduc(end:-1:1);
behav_predict_obj_reduc = behav_predict_obj_reduc(:,end:-1:1);
behav_predict_acc_reduc_ci95 = behav_predict_acc_reduc_ci95(end:-1:1);

cutoff95 = (0.95*behav_predict_acc-100/3)+100/3;
cutoff99 = (0.99*behav_predict_acc-100/3)+100/3;

mindim = find(behav_predict_acc_reduc>cutoff95,1,'first')-1; % -1 because we need to start counting at 0
maxdim = find(behav_predict_acc_reduc<cutoff99,1,'last')-1;
fprintf('We need between %i and %i dimensions to reach 95-99%% performance in predicting individual trials.\n',mindim,maxdim)

% nc1 = noise_ceiling+noise_ceiling_ci95;
% nc2 = noise_ceiling-noise_ceiling_ci95;
% y2 = [nc1 nc1 nc2 nc2];
% hc = patch(x,y2,[0.7 0.7 0.7]);
% hc.EdgeColor = 'none';


% compare similarity matrices
for i_dim = 1:49
    r_reduc(i_dim) = corr(squareformq(spose_sim),squareformq(spose_sim_reduc{i_dim}));
end

% reverse r_reduc
r_reduc = r_reduc(end:-1:1);

mindim2 = find(r_reduc.^2>0.95,1,'first')-1; % -1 because we need to start counting at 0
maxdim2 = find(r_reduc.^2<0.99,1,'last')-1;
fprintf('We need between %i and %i dimensions to explain 95-99%% variance in similarity.\n',mindim2,maxdim2)

% TODO: plot
% figure, plot(48:-1:0,100*(behav_predict_acc_reduc-100/3)/(behav_predict_acc-100/3)) % subtract chance
% addpath(fullfile(matlab_dir,'shadedErrorBar'))
figure('Position',[0 1200 1200 550])
subtightplot(1,2,1)
% shadedErrorBar(0:49,[behav_predict_acc_reduc behav_predict_acc],[behav_predict_acc_reduc_ci95' behav_predict_acc_ci95])
% 95% means not 95% of that accuracy, but 95% of the performance relative to chance
cutoff95 = (0.95*behav_predict_acc-100/3)+100/3;
cutoff99 = (0.99*behav_predict_acc-100/3)+100/3;
% also, add noise ceiling
hold on
x = [0 50 50 0];
y = [cutoff95 cutoff95 cutoff99 cutoff99];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hc.FaceAlpha = 0.3;
x = [mindim maxdim maxdim mindim];
y = [70 70 0 0];
hcb = patch(x,y,[138 204 101]/255);
hcb.EdgeColor = 'none';
hcb.FaceAlpha = 0.5;
plot(0:49,[behav_predict_acc_reduc behav_predict_acc],'k','LineWidth',3)
plot([0 49],[100/3 100/3],'k--')
xlim([0 49])
ylim([0 70])
xlabel('Number of dimensions retained')
ylabel('Accuracy')


subtightplot(1,2,2)

% 95% means here means 95% and 99% means 99%
hold on
x = [0 50 50 0];
y = [95 95 99 99];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hc.FaceAlpha = 0.3;
x = [mindim2 maxdim2 maxdim2 mindim2];
y = [108.2 108.2 0 0];
hcb = patch(x,y,[138 204 101]/255);
hcb.EdgeColor = 'none';
hcb.FaceAlpha = 0.5;
xlim([0 49])
ylim([0 108.2])
plot(0:49,[100*r_reduc.^2 100],'k','LineWidth',3)
xlabel('Number of dimensions retained')
ylabel('Variance explained')


% x-axis: number of dimensions retained per object
% y-axis: variance explained
% add dashes to where 95% and 99% is
if dosave
    saveas(gcf,'Fig6.svg')
end


%% Figure 3: Show examples (loading high quality images)

dosave = 0;

% in original figures, we used high quality images, here for space reasons
% we use low quality (original code still below but commented)

% sort each dimension
[~,dimsortind] = sort(spose_embedding49,'descend');

% now get word cloud data
for i_dim = 1:49
    str = strings;
    % extend responses at comma, remove leading and trailing spaces
    for i_sub = 1:20
        s = strsplit(dimlabel_answers{i_sub,i_dim},',');
        for k = 1:length(s)
            s{k} = strtrim(s{k});
        end
        str(end+1:end+length(s)) = string(s);
    end
    str(1) = [];
    str = str';
    [dimlabel_words{i_dim},~,idx] = unique(str);
    dimlabel_num_occur{i_dim} = histcounts(idx,numel(dimlabel_words{i_dim}));
    
end


n_im = 8;
for i_dim = [3 11 12 15 17 40] % chosen dimensions
    
    figure('Position',[870 2043 2476 310],'color','none')
    
    % plot word cloud
    subtightplot(1,9,1,0.005)
    ha = wordcloud(dimlabel_words{i_dim},dimlabel_num_occur{i_dim});
    ha.Color = [0 0 0];
    ha.HighlightColor = colors(i_dim,:);
%     title(['dim ' num2str(i)]);
    ht.FontSize = 20;
    
    for i = 1:n_im
        
        subtightplot(1,9,i+1,0.005)
%         currfn = fullfn{strcmp(fn,[unique_id{dimsortind(i,i_dim)} '.jpg'])};
%         img = imread(currfn);
        imagesc(im{dimsortind(i,i_dim)})
        axis off square
        
    end
    

    
    if dosave
        hf = gcf;
        hf.Renderer = 'painters';
        % print(sprintf('tempim_%02i.pdf',i_dim),'-dpdf','-bestfit')
        saveas(hf,sprintf('tempim_%02i.svg',i_dim))
    end
    
end


% % Original code:
% 
% % get image paths for top images of all dimensions
% fullfn = cell(1854,1);
% fn = dir(fullfile(base_dir,'reference_images/*.jpg');
% for i = 1:length(fn)
%     fullfn{i,1} = fullfile(base_dir,'reference_images',fn(i).name);
% end
% fn = {fn.name}';
% 
% % get sortind again (Mac sorts differently to Matlab)
% load(fullfile(base_dir,'sortind.mat'));
% 
% % re-sort
% fn = fn(sortind);
% fullfn = fullfn(sortind);
% 
% % now sort each dimension
% [~,dimsortind] = sort(spose_embedding49,'descend');
% 
% % now get word cloud data
% for i_dim = 1:49
%     str = strings;
%     % extend responses at comma, remove leading and trailing spaces
%     for i_sub = 1:20
%         s = strsplit(dimlabel_answers{i_sub,i_dim},',');
%         for k = 1:length(s)
%             s{k} = strtrim(s{k});
%         end
%         str(end+1:end+length(s)) = string(s);
%     end
%     str(1) = [];
%     str = str';
%     [dimlabel_words{i_dim},~,idx] = unique(str);
%     dimlabel_num_occur{i_dim} = histcounts(idx,numel(dimlabel_words{i_dim}));
%     
% end
% 
% 
% n_im = 8;
% for i_dim = [3 11 12 15 17 40]
%     
%     figure('Position',[870 2043 2476 310],'color','none')
%     
%     % plot word cloud
%     subtightplot(1,9,1,0.005)
%     ha = wordcloud(dimlabel_words{i_dim},dimlabel_num_occur{i_dim});
%     ha.Color = [0 0 0];
%     ha.HighlightColor = colors(i_dim,:);
% %     title(['dim ' num2str(i)]);
%     ht.FontSize = 20;
%     
%     for i = 1:n_im
%         
%         subtightplot(1,9,i+1,0.005)
%         currfn = fullfn{strcmp(fn,[unique_id{dimsortind(i,i_dim)} '.jpg'])};
%         img = imread(currfn);
%         imagesc(img)
%         axis off square
%         
%     end
%     
% 
%     
%     hf = gcf;
%     hf.Renderer = 'painters';
%     % print(sprintf('tempim_%02i.pdf',i_dim),'-dpdf','-bestfit')
%     saveas(hf,sprintf('tempim_%02i.svg',i_dim))
%     
% end

%% Show all dimensions Extended Data Figure 2, but without images (for code with images, see commented code below)

dosave = 0;

hf = figure('Position',[870 2043 1200 1200]);

for i_dim = 1:49
    subtightplot(7,7,i_dim)
    
    
    % plot word cloud
    ha = wordcloud(dimlabel_words{i_dim},dimlabel_num_occur{i_dim});
    ha.Color = [0 0 0];
    ha.HighlightColor = colors(i_dim,:);
    
    title(sprintf('Dimension %i: %s',i_dim,labels{i_dim}))
end
if dosave
    hf.Renderer = 'painters';
    print(sprintf('tempim_%02i.svg',i_dim),'-dpdf','-bestfit')
end
close(hf)



%% Figure 4: Make crystal plot, with an example plot for abacus

dosave = 0;

% First, get 2d MDS solution
rng(42) % use fixed random number generator
[Y2,stress] = mdscale(dissim,2,'criterion','metricstress');

% Next, to visualize how tsne is run, we set clusters according to the
% strongest dimension in an object
[~,clustid] = max(spose_embedding49,[],2);

% Then, based on this solution, initialize t-sne solution with multiple
% perplexities in parallel (multiscale)
rng(1)
perplexity1 = 5; perplexity2 = 30;
D = dissim / max(dissim(:));
P = 1/2 * (d2p(D, perplexity1, 1e-5) + d2p(D, perplexity2, 1e-5)); % convert distance to affinity matrix using perplexity
figure
colormap(colors)
Ytsne = tsne_p(P,clustid,Y2);

% if interested plot words
figure
text(Ytsne(:,1),Ytsne(:,2),words)
xlim(1.05*[min(Ytsne(:,1)) max(Ytsne(:,1))])
ylim(1.05*[min(Ytsne(:,2)) max(Ytsne(:,2))])

% points within a specific polygon
% ff = [28;29;36;39;46;51;53;60;72;79;84;93;94;99;100;119;160;172;180;190;192;194;204;208;217;222;224;252;259;261;286;300;329;337;340;355;357;358;364;388;394;412;432;433;445;448;454;464;468;470;474;475;487;509;510;511;521;524;529;549;554;567;568;584;593;603;607;614;617;630;631;635;638;644;653;654;666;714;722;728;730;742;747;749;752;755;760;780;785;788;794;795;800;801;823;830;838;859;870;872;880;882;883;884;887;888;893;894;897;900;901;908;920;926;941;954;955;957;963;973;982;997;1020;1030;1047;1048;1049;1056;1063;1068;1096;1137;1139;1142;1143;1144;1145;1148;1156;1167;1180;1192;1202;1205;1208;1211;1214;1216;1219;1223;1228;1238;1246;1253;1256;1269;1278;1282;1287;1292;1305;1306;1312;1315;1325;1333;1336;1338;1342;1367;1370;1371;1375;1380;1382;1388;1391;1392;1400;1413;1414;1418;1425;1426;1433;1442;1462;1468;1489;1495;1507;1509;1511;1520;1534;1539;1540;1568;1572;1590;1603;1624;1628;1640;1643;1675;1677;1678;1713;1714;1720;1729;1735;1740;1766;1768;1781;1810;1812;1830;1836;1842;1848];
% ff = [4;14;15;31;32;40;41;43;44;64;65;76;82;103;113;115;118;122;134;136;150;157;165;174;202;207;228;233;236;264;278;282;289;292;317;326;338;348;349;350;366;367;376;379;413;417;418;422;431;458;471;497;547;551;552;578;582;619;620;646;659;674;697;699;702;703;704;705;706;713;720;774;835;839;861;877;905;907;911;915;921;931;952;967;975;990;995;1005;1008;1014;1023;1038;1041;1067;1075;1076;1078;1079;1080;1082;1099;1103;1112;1123;1129;1130;1132;1134;1135;1136;1151;1152;1157;1159;1168;1181;1182;1183;1184;1189;1195;1204;1210;1220;1225;1232;1236;1237;1245;1248;1254;1264;1275;1281;1285;1308;1324;1335;1349;1373;1402;1406;1419;1420;1498;1516;1518;1527;1541;1548;1558;1571;1585;1596;1619;1645;1651;1673;1709;1727;1733;1755;1779;1794;1796;1803;1806;1840;1851;1854];
% ff = [5;74;98;128;199;241;245;247;248;249;280;281;284;294;309;334;395;396;403;406;408;439;478;483;492;494;525;587;611;612;641;753;767;770;773;833;898;927;928;958;996;999;1000;1001;1012;1025;1028;1072;1089;1090;1128;1138;1162;1164;1203;1218;1241;1243;1301;1303;1337;1376;1389;1404;1408;1438;1447;1464;1550;1565;1611;1625;1642;1649;1667;1668;1692;1706;1715;1742;1751;1758;1762;1763;1772;1785;1792;1800];
ff = [14;15;32;40;41;43;64;65;82;103;113;118;136;157;165;202;207;228;233;236;264;282;289;292;338;348;367;413;417;458;551;578;659;674;697;703;705;706;713;720;774;835;839;861;877;905;907;911;921;931;975;1005;1008;1023;1038;1041;1067;1075;1076;1080;1103;1112;1123;1129;1134;1136;1152;1157;1168;1181;1182;1183;1189;1195;1248;1275;1308;1349;1373;1406;1516;1527;1541;1548;1558;1596;1645;1709;1733;1755;1803;1806;1854];

% Now add the "crystals", i.e. rose plots

v = zeros(200000,2);
f = zeros(100000,3);
ct = 0;
cnt1 = 0;
cnt2 = 0;

scaling = 2.8;

for ii = 1:1854
    if ii == 1854, fprintf('\n'), end
    [x,y] = pol2cart(repmat(linspace(0,2*pi,49+1),[49 1]),scaling*repmat(spose_embedding49(ii,:)',[1 49+1]));
    for i = 1:49
        ct = ct+1;
        v(cnt1+1:cnt1+3,:) = [Ytsne(ii,1) Ytsne(ii,2); x(i,i)+Ytsne(ii,1) y(i,i)+Ytsne(ii,2); x(i,i+1)+Ytsne(ii,1) y(i,i+1)+Ytsne(ii,2)];
        f(cnt2+1,:) = ((ct-1)*3 + (1:3));
        cnt1 = cnt1+3;
        cnt2 = cnt2+1;
    end
end
v(cnt1+1:end,:) = [];
f(cnt2+1:end,:) = [];

hf = figure;
hf.Position = [-393        1421         946         905];
patch('faces',f,'vertices',v,'FaceVertexCData',repmat(linspace(0,1,49),1,1854)','FaceColor','flat','edgecolor','none','facealpha',0.85)
colormap(colors)

axis equal off tight

if dosave
    saveas('Fig4a.svg')
    % for high-quality figure, download fig2svg from file exchange
%     fig2svg('Fig4a.svg')
end
    

% % Figure out where different objects are
% currwords = {'pants','hand','fish','meerkat','zucchini','paint','pancake','wood','hammer','laptop','taxi','piano'};
% [~,~,ii] = intersect(currwords,words,'stable');
% hold on
% for i = 1:length(ii)
%     plot(Ytsne(ii(i),1),Ytsne(ii(i),2),'o','MarkerFaceColor',[0 0 0],'MarkerEdgeColor','none','MarkerSize',5)
%     text(Ytsne(ii(i),1),Ytsne(ii(i),2),currwords(i))
% end

%% classification analysis (no figure involved but included for reproducibility)

% results printed to screen
predict_category(base_dir);

%% Figure 5: Now plot one example where we have zoomed in (microscope, word = 1000; bottle, word = 171; squid, word = 1529)

for i_example = [2 171 351 601 745 898 923 1000 1062 1131 1166 1198 1259 1284 1321 1529 1577 1787]
    
    
    v0 = [];
    f0 = [];
    v1 = [];
    ct = 0;
    [x0,y0] = pol2cart(repmat(linspace(0,2*pi,49+1),[49 1]),scaling*repmat(spose_embedding49(i_example,:)',[1 49+1]));
    [th,r] = cart2pol(x0,y0); [x1,y1] = pol2cart(th,r-0.05);
    for i = 1:49
        ct = ct+1;
        v0 = [v0; [0 0; x0(i,i) y0(i,i); x0(i,i+1) y0(i,i+1)]];
        f0 = [f0; ((ct-1)*3 + (1:3))];
        v1 = [v1; [0 0; x1(i,i) y1(i,i); x1(i,i+1) y1(i,i+1)]];
    end
    
    figure('Position',[1 103 1200 1200])
    patch('faces',f0,'vertices',v0,'FaceVertexCData',linspace(0,1,49)','FaceColor','flat','edgecolor','none','facealpha',0.5)
    colormap(colors)
    axis off square equal tight
    
    
    hold on
    clear ht rd
    rot = linspace(0,2*pi,49+1);
    rot = conv(rot,[0.5 0.5]);
    rot = rot(2:end-1);
    for i = 1:49
        if r(i,1)<1.5, continue, end
        vind = 3*(i-1);
        ht(i) = text(mean(v1(vind+(2:3),1)),mean(v1(vind+(2:3),2)),labels_short{i});
        ht(i).Rotation = rad2deg(rot(i));
        rd(i) = mod(rad2deg(rot(i)),180);
        ht(i).FontSize = 18; % was 18
        ht(i).FontName = 'Avenir Next';
        ht(i).HorizontalAlignment = 'right';
        if ht(i).Rotation>90 && ht(i).Rotation<270
            ht(i).Rotation = ht(i).Rotation+180;
            ht(i).HorizontalAlignment = 'left';
        end
    end
    title(words{i_example})
    
    if dosave
        fig2svg(sprintf('Fig4c%i.svg',i_example))
    end

    
end


%% Figure 7: Predictions of human typicality

dosave = 0;

% Explanation of relevant variables
% categories27: category names for the 27 categories (alphabetically sorted)
% category27_typicality_rating_normed: typicality ratings for objects of the 27 categories (normed within each participant to make scale use more comparable)
% category27_ind: which of the 1,854 objects belong to each of the 27 categories
% category27_subind: which of the 27 categories do we use
% best_match27: which of the 49 dimensions best matches to the 27 categories (if any)



% to show relationship between categories and labels
% [categories27(category27_subind); labels(best_match27(category27_subind))]';


% extract relevant parts of embedding and typicalities
for i = 1:length(category27_subind)
    typicality_normed{i} = category27_typicality_rating_normed{category27_subind(i)};
    spose{i} = spose_embedding49(category27_ind{category27_subind(i)},best_match27(category27_subind(i)));
end

for i = 1:length(category27_subind)
    % using the unsorted one for both makes it easiest
    [r_typicality_s(i,1),p_typicality_s(i,1)] = corr(category27_typicality_rating_normed{category27_subind(i)}, spose_embedding49(category27_ind{category27_subind(i)},best_match27(category27_subind(i))),'tail','right','type','spearman');
end

[~,~,~,p_typicality_s_adjusted] = fdr_bh(p_typicality_s);
    
% Get typicality colors
typicality_colors = best_match27(category27_subind);
% sort typicality by size of correlation
[~,sortindtmp] = sort(r_typicality_s,'descend');
spose = spose(sortindtmp);
typicality_normed = typicality_normed(sortindtmp);
typicality_colors = typicality_colors(sortindtmp);

hf = figure('Position',[578 426 1674 703]);
clear ht hx hy
for i = 1:17
    subtightplot(3,6,i,0.05)
    if i <= 17
        plot(spose{i},typicality_normed{i},'o','MarkerFaceColor',colors(typicality_colors(i),:),'MarkerEdgeColor','none','MarkerSize',8)
    else
        plot(spose{i},typicality_normed{i},'o','MarkerFaceColor',[0.2 0.2 0.9],'MarkerEdgeColor','none','MarkerSize',8)
    end
    
    deltax = range(spose{i}); deltay = range(typicality_normed{i});
%     text(mean(spose{i}),mean(typicality_normed{i}),sprintf('%.2f',corr(spose{i}',typicality_normed{i}','type','spearman')))
    ht(i) = text(max(spose{i})-0.05*deltax,min(typicality_normed{i}+0.1*deltax),sprintf('%s = %.2f','\rho',r_typicality_s(sortindtmp(i))),'HorizontalAlignment','right');
    xlim([min(spose{i})-0.08*deltax max(spose{i})+0.08*deltax])
    ylim([min(typicality_normed{i})-0.08*deltay max(typicality_normed{i})+0.08*deltay])
    axis square
    ax = gca;
    ax.XTick = []; ax.YTick = [];
    hy(i) = ylabel(categories27{category27_subind((sortindtmp(i)))});
    hx(i) = xlabel(labels_short{best_match27(category27_subind(sortindtmp(i)))});
end

subtightplot(3,6,18,0.05)
ht(end+1) = text(0.9,0.05,'Spearman''s \rho','HorizontalAlignment','right');
% plot(0.5,0.5,'o','MarkerEdgeColor','none')
xlim([0 1]), ylim([0 1])
axis square
ax = gca;
ax.XTick = []; ax.YTick = [];
hy(end+1) = ylabel('Typicality scale');
hx(end+1) = xlabel('Dimension');

set([ht hx hy],'FontName','Myriad Pro','FontSize',16,'Color',[0 0 0])
set([hx hy],'HorizontalAlignment','center')

set(ht(p_typicality_s_adjusted(sortindtmp)<0.05),'FontWeight','bold')

if dosave
    saveas(hf,'Figure6.svg')
end

% Bootstrap (uncorrected) confidence intervals
rng(1)
for i = 1:length(category27_subind)
    % using the unsorted one for both makes it easiest
    c1 = category27_typicality_rating_normed{category27_subind(i)};
    c2 = spose_embedding49(category27_ind{category27_subind(i)},best_match27(category27_subind(i)));
    nc = length(c1);
    rnd = randi(nc,nc,1000);
    for j = 1:1000
    r_typicality_s_boot(i,j) = corr(c1(rnd(:,j)), c2(rnd(:,j)) ,'type','spearman');
    end
end
r_typicality_s_ci95_lower = tanh(atanh(r_typicality_s) - 1.645*std(atanh(r_typicality_s_boot),[],2)); % reflects one-sided 95% CI (still unsorted)
r_typicality_s_ci95_upper = tanh(atanh(r_typicality_s) + 1.645*std(atanh(r_typicality_s_boot),[],2)); % reflects one-sided 95% CI

%% Figure 8: Get human predictions and compare to model

dosave = 0;

object_names20 = {'bazooka', 'bib', 'crowbar', 'crumb', 'flamingo', 'handbrake', 'hearse', 'keyhole', 'palm_tree', 'scallion', 'sleeping_bag', ...
    'spider_web', 'splinter', 'staple_gun', 'suitcase', 'syringe', 'tennis_ball', 'woman', 'workbench', 'wreck'};

n_target = 20;

ind = zeros(n_target,1);
for i_target = 1:n_target
    ind(i_target) = find(strcmp(unique_id,object_names20{i_target}));
end

Rt = mean(ratings_translated_all,3);
% we need to adjust the scaling because we later subtract the minimum
minRt = min(Rt);
mRt = mean(Rt);
Rt = Rt - mRt;
Rt = (1+minRt).*Rt;
Rt = Rt+mRt;

spose_sub = spose_embedding49;
spose_sub(ind,:) = Rt - min(Rt); % remove minimum again (otherwise all objects will end up being highly similar to each other)

cp = embedding2sim(spose_sub);

true_sim = spose_sim(ind,ind);
predicted_sim = cp(ind,ind);

% add structure using clustering with arbitrary cluster number
d = squareformq(1-true_sim)';
z = linkage(d,'centroid');
% dendrogram(z)
[~,ind2] = sort(cluster(z,'maxclust',8));


h = figure;
h.Position(3:4) = [1200 768];
ha = subtightplot(1,3,1);
imagesc(predicted_sim(ind2,ind2),[0 1])
colormap(viridis)
hold on
text(24,-4,'Similarity matrix (dimension ratings)','HorizontalAlignment','center','FontSize',16);
axis off square tight
hb = subtightplot(1,3,2);
imagesc(true_sim(ind2,ind2),[0 1])
text(24,-4,'Similarity matrix (reference)','HorizontalAlignment','center','FontSize',16);
colormap(viridis)
axis off square tight
hc = subtightplot(1,3,3);
hc.Position(1) = hc.Position(1)+0.02;
% plot(squareformq(spose_sim48),squareformq(1-RDM48_triplet),'o','MarkerFaceColor',[0.1651 0.4674 0.5581],'MarkerEdgeColor','none','MarkerSize',3);
plot(squareformq(predicted_sim),squareformq(true_sim),'o','MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor','none','MarkerSize',3);
hold on
plot([0 1],[0 1],'k')
axis square tight
xlabel('similarity from ratings')
ylabel('similarity (reference)')
legend('R = 0.85','Location','NorthWest')
if dosave
    saveas(h,'Figure8.svg')
end

% do stats (randomization test, shuffling labels)
rng(42)
n_shuffle = 10000; % tried it with 1e6, still not a single time exceeded
[~,randind] = sort(rand(20,n_shuffle));
true_sim_vec = squareformq(true_sim);
r_reference = corr(squareformq(predicted_sim),squareformq(true_sim));
r_randomization = zeros(n_shuffle,1);
for i = 1:n_shuffle
   r_randomization(i,1) = corr(squareformq(predicted_sim(randind(:,i),randind(:,i))),true_sim_vec);
end

% get 95% confidence interval (cannot be across objects since otherwise it's biased)
rng(1)
rnd = randi(nchoosek(20,2),nchoosek(20,2),1000);
c1 = squareformq(predicted_sim);
c2 = squareformq(true_sim);
for i = 1:1000
   r20_boot(i) = corr(c1(rnd(:,i)),c2(rnd(:,i)));
end
r20_ci95_lower = tanh(atanh(corr(c1,c2)) - 1.96*std(atanh(r20_boot),[],2));
r20_ci95_upper = tanh(atanh(corr(c1,c2)) + 1.96*std(atanh(r20_boot),[],2));

%% Make Extended Data Figure 1 (consistency of dimensions)

dosave = 0;

load(fullfile(variable_dir,'sortind.mat')); % need this because original order is wrong
refdir = fullfile(base_dir,'reference_models');
for i_model = 1:20
    fn = dir(fullfile(refdir,sprintf('s%02i',i_model),'*.txt'));
    fn = fullfile(fn(end).folder,fn(end).name);
    tmp = load(fn);
    % remove empty dimensions
    tmp2 = tmp(:,any(tmp>0.1));
    reference_models{i_model,1} = tmp2(sortind,:);
    n_dim_reference(i_model) = size(reference_models{i_model},2);
end

% Correlate dimensions (this slightly overestimates the performance, given
% that each dimension can be picked several times, but there is no other
% way - otherwise some dimensions would go unmatched)
for i_model = 1:20
    reproducibility(:,i_model) = max(corr(spose_embedding49,reference_models{i_model}),[],2);
end

% test split-half prediction
for i_model = 1:20
    [~,maxind(:,i_model)] = max(corr(spose_embedding49(1:2:end,:),reference_models{i_model}(1:2:end,:)),[],2);
    [~,maxind2(:,i_model)] = max(corr(spose_embedding49(2:2:end,:),reference_models{i_model}(2:2:end,:)),[],2);
    c1 = corr(spose_embedding49(1:2:end,:),reference_models{i_model}(1:2:end,:));
    c2 = corr(spose_embedding49(2:2:end,:),reference_models{i_model}(2:2:end,:));
    for i = 1:49, tmp1(i,i_model) = c1(i,maxind2(i,i_model)); tmp2(i,i_model) = c2(i,maxind(i,i_model)); end
end

% fisher-z convert before averaging across models
mean_reproducibility = mean(atanh(reproducibility),2);
reproducibility_ci95 = 1.96*std(atanh(reproducibility),[],2)/sqrt(20);

% for plotting, the upper bound will be mean + 95% CI, then conversion back
% to correlation, same for lower bound
upper_bound = tanh(mean_reproducibility+reproducibility_ci95);
lower_bound = tanh(mean_reproducibility-reproducibility_ci95);
% now update mean reproducibility, as well
mean_reproducibility = tanh(mean_reproducibility);

hf = figure;
hf.Position(3:4) = [1024 768];
hold on
x = [1:49 49:-1:1];
y = [lower_bound' upper_bound(end:-1:1)'];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
plot(mean_reproducibility,'k','linewidth',1)
plot(reproducibility,'o','MarkerFaceColor',[0 0 0],'MarkerEdgeColor','none','MarkerSize',3)
ylim([0 1])
xlim([0 50])

if dosave
    saveas(hf,'Supplement1.svg')
end

% Test correlation between rank of reliability and dimension number
[~,reproducibility_ind] = sort(mean_reproducibility,'descend');
r_rank = corr((1:49)',reproducibility_ind);

% run 100000 permutations
rng(1)
[~,perm] = sort(rand(49,100000));
r_rank_perm = corr(perm,reproducibility_ind);
% is obviously never exceeded (smaller sign because it's a negative correlation)
p = mean([r_rank_perm;r_rank] >= r_rank);

% run 1000 bootstrap samples for confidence intervals
rng(2)
rnd = randi(49,49,1000);
for i = 1:1000
    r_rank_boot(:,i) = corr(rnd(:,i),reproducibility_ind(rnd(:,i)));
end
r_rank_ci95_lower = tanh(atanh(r_rank) - 1.96*std(atanh(r_rank_boot),[],2)); % reflects 95% CI
r_rank_ci95_upper = tanh(atanh(r_rank) + 1.96*std(atanh(r_rank_boot),[],2)); % reflects 95% CI