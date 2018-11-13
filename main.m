%% Load and modify the network
net = googlenet;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
newLayers = [
    fullyConnectedLayer(3,'Name','fc(3)','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    angularRegressionLayerL2('regression with angular loss')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc(3)');

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])
%% Load data (ColorChecker)
list = dir('./Reprocessed/*.tiff');
for i = 1:length(list)
    I = imread(strcat(list(i).folder,'\',list(i).name));
    
    if (size(I,1) < size(I,2))
        I = imresize(I,[224 NaN]);
        I = I(:,57:280,:);
    else
        I = imresize(I,[NaN 224]);
        I = I(57:280,:,:);
    end
    imgs(:,:,:,i) = I;
end
imgs = double(imgs);
imgs = (imgs - mean(imgs,4)) ./ std(imgs,0,4);

response = load('./Reprocessed/illuminantsNormalized.mat');
response = response.illuminants;
%% Load data (SFU Grayball)
% fID = fopen('Source_Image/file.lst');
% list = textscan(fID,'%s','delimiter','\n'); 
% list = [list{1,1}];
% fclose(fID);
% 
% for i = 1:size(list,1)
%     imgs(:,:,:,i) = imread(list{i,1});
% end
% imgs = imgs(9:232,1:224,:,:);
% 
% response = load('Source_Image/real_illum_11346_Normalized.mat');
% response = response.real_rgb;

%% Divide training and testing sets (without cross-validation)
idx = randperm(size(imgs,4),round(size(imgs,4)/15));
imgsTest = imgs(:,:,:,idx);
imgs(:,:,:,idx) = [];
responseTest = response(idx,:);
response(idx,:) = [];

%% Data Augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',[-30 30], ...
    'RandYTranslation',[-30 30]);
inSize = net.Layers(1).InputSize;
auimds = augmentedImageSource(inSize(1:2),imgs,response,...
     'DataAugmentation',imageAugmenter);
%% Training
options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',300, ...  
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',120,...
    'Shuffle','every-epoch',...
    'Verbose',false ,...
    'Plots','training-progress');

mynet548 = trainNetwork(auimds,lgraph,options);
%% Evauation

%single-image inference
%predict(mynet,imgs(:,:,:,1))

responsePredicted = predict(mynet548,imgsTest,'ExecutionEnvironment','cpu');

for j = 1:size(responseTest,1)
    e1 = responseTest(j,:); 
    e2 = responsePredicted(j,:);
    angles(j,:) = rad2deg(acos( (e1*e2')/norm(e1)/norm(e2) ));
end

summary = table(mean(angles),median(angles),std(angles), ...
    'VariableNames',{'mean', 'median', 'std'})