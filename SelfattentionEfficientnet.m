clear all
clc

%% Import Training Dataset to Matlab
imagefolder='D:\Drive 1\LBC cervical cancer dataset';
imdsTrain = imageDatastore(imagefolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

% distribution of images in each class
tbl = countEachLabel(imdsTrain);
classNamesnew = categories(imdsTrain.Labels);
numClasses = numel(categories(imdsTrain.Labels));

%% Balancing the dataset

% Determine the smallest amount of images in a category
%minSetCount = min(tbl{:,2}); 

% Limit the number of images to reduce the time it takes
% run this example.
%maxNumImages = 100;
%minSetCount = min(maxNumImages,minSetCount);
%% split the dataset for training and testing
% Use splitEachLabel method to trim the set.
%imds = splitEachLabel(imds, minSetCount, 'randomize');
%[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');
[imdsTrain,imdsValidation] = splitEachLabel(imdsTrain, 0.7);
% Notice that each set now has exactly the same number of images.


%% Import Testing Dataset to Matlab
% imagefolder2='D:\Drive 1\Tomato\tomato\val';
% imdsValidation = imageDatastore(imagefolder2, 'LabelSource', 'foldernames', 'IncludeSubfolders',true,'FileExtensions','.jpg');
% 
% % distribution of images in each class
% tbl2 = countEachLabel(imdsValidation);

%% Load the network
% Load pretrained network
%net = resnet50();
%net=darknet19();
%net=densenet201();
%net = nasnetmobile();
%net=squeezenet();
%net=resnet18();
%net=resnet101();
%net=efficientnetb0;
%net=alexnet();
%net=googlenet();
%net=inceptionv3();
%net=shufflenet();
%net=inceptionresnetv2();
%net=darknet53();
%net=xception();
%net=mobilenetv2();


%% Resize images 
% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.
imageSize = net.Layers(1).InputSize;
augmentedTrainSet = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing','gray2rgb');
augmentedValidationSet = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing','gray2rgb');

%% transfer Learning
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
   lgraph = layerGraph(net);
end 
% 
% [learnableLayer,classLayer] = findLayersToReplace(lgraph);
% 
% numClasses = numel(categories(imdsTrain.Labels));
% 
% if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
%     newLearnableLayer = fullyConnectedLayer(numClasses, ...
%         'Name','new_fc', ...
%         'WeightLearnRateFactor',10, ...
%         'BiasLearnRateFactor',10);
% 
% elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
%     newLearnableLayer = convolution2dLayer(1,numClasses, ...
%         'Name','new_conv', ...
%         'WeightLearnRateFactor',10, ...
%         'BiasLearnRateFactor',10);
% end
% 
% lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
% 
% newClassLayer = classificationLayer('Name','new_classoutput');
% lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])
%% to run efficientnet
layer = fullyConnectedLayer(numClasses,Name="efficientnet-b0|model|head|dense|MatMul");
lgraph = replaceLayer(lgraph,"efficientnet-b0|model|head|dense|MatMul",layer);
layer2=classificationLayer('Classes',classNamesnew );

lgraph = replaceLayer(lgraph,"classification",layer2);

%% remove layers and add self attention
lgraph = removeLayers(lgraph, ["efficientnet-b0|model|head|dense|MatMul" "Softmax" "classoutput"]);
layers=[flattenLayer('Name', 'flatten')
    selfAttentionLayer(8, 1280, 'Name', 'self_attention');
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool','flatten');
%% Perform Augmentation
pixelRange = [-50 50];
scaleRange = [0.5 2.5];
shearRange=[-60 60];
    imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation',[-70,70],...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXShear',shearRange,...
    'RandXShear',shearRange,...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange....
         );
augimdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);
%% Resize test Images
augimdsValidation = augmentedImageDatastore(imageSize,imdsValidation,'ColorPreprocessing','gray2rgb');

%% train the network
miniBatchSize = 5;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

%% %% Classification using deep learning
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);

%% Extract Features From the layer just before classification
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net)

%layer = 'pool5';
%layer = 'avg_pool';
%layer='new_fc';
%layer='global_average_pooling2d_1';
%layer='relu_conv10';
%layer='pool5-7x7_s1';
%layer='avg1';
%layer='node_200';
%layer='new_conv';
%layer='efficientnet-b0|model|head|dense|MatMul';
layer='efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool';
%layer='self_attention';
%layer='pool10';
featuresTrain = activations(net,augimdsTrain,layer,'MiniBatchSize', 10, 'OutputAs','rows');
featuresTest = activations(net,augimdsValidation,layer,'MiniBatchSize', 10,'OutputAs','rows');

whos featuresTrain

%% train an SVM classifier
% Get training labels from the trainingSet
trainingLabels = imdsTrain.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(featuresTrain', trainingLabels', ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% test the SVM classifier

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, featuresTest', 'ObservationsIn', 'columns');

% Get the known labels
testLabels = imdsValidation.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% Display the mean accuracy

%layer = 'avg_pool';
%layer='new_fc';
%layer='global_average_pooling2d_1';
%layer='relu_conv10';
%layer='pool5-7x7_s1';
%layer='avg1';
%layer='node_200';
%layer='pool5';
featuresTrain = activations(net,augimdsTrain,layer,'MiniBatchSize', 10, 'OutputAs','rows');
featuresTest = activations(net,augimdsValidation,layer,'MiniBatchSize', 10,'OutputAs','rows');

whos featuresTrain

%% train an SVM classifier
% Get training labels from the trainingSet
trainingLabels = imdsTrain.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(featuresTrain', trainingLabels', ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% test the SVM classifier

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, featuresTest', 'ObservationsIn', 'columns');

% Get the known labels
testLabels = imdsValidation.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% Display the mean accuracy
mean(diag(confMat))
D1=[featuresTrain;featuresTest];
labelD1=[trainingLabels;testLabels];
dataROP=table(D1,labelD1);
