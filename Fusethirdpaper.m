clear all
clc

load FeaturesTSelfattentionEffcientNet_Pool_SKY
load FeaturesTSelfattentionMobileNet_Pool_SKY
load FeaturesTSelfattentionResNet18_Pool_SKY

load FeaturesTSelfattentionEffcientNet_Self_SKY
load FeaturesTSelfattentionMobileNet_Self_SKY
load FeaturesTSelfattentionResNet18_Self_SKY

%% 
Fuse_Efficient_SKY=[FeaturesTSelfattentionEffcientNet_Self_SKY,FeaturesTSelfattentionEffcientNet_Pool_SKY];
DWT_Fuse_Efficient_SKY=dwt_deepfeatures(Fuse_Efficient_SKY);
DWT_Fuse_Efficient_SKY_Table=table(DWT_Fuse_Efficient_SKY',labels);
%% 
Fuse_Mobile_SKY=[FeaturesTSelfattentionMobileNet_Self_SKY,FeaturesTSelfattentionMobileNet_Pool_SKY];
DWT_Fuse_Mobile_SKY=dwt_deepfeatures(Fuse_Mobile_SKY);
DWT_Fuse_Mobile_SKY_Table=table(DWT_Fuse_Mobile_SKY',labels);

%% 
Fuse_ResNet18_SKY=[FeaturesTSelfattentionResNet18_Self_SKY,FeaturesTSelfattentionResNet18_Pool_SKY];
DWT_Fuse_ResNet18_SKY=dwt_deepfeatures(Fuse_ResNet18_SKY);
DWT_Fuse_ResNet18_SKY_Table=table(DWT_Fuse_ResNet18_SKY',labels);

%%
DWT_ALL=[DWT_Fuse_Efficient_SKY',DWT_Fuse_Mobile_SKY',DWT_Fuse_ResNet18_SKY'];
DWT_ALL_SKY=table(DWT_ALL,labels);

%%
clear all
clc
load FeaturesTSelfattentionEffcientNet_Pool_LBC
load FeaturesTSelfattentionMobileNet_Pool_LBC
load FeaturesTSelfattentionResNet18_Pool_LBC

load FeaturesTSelfattentionEffcientNet_Self_LBC
load FeaturesTSelfattentionMobileNet_Self_LBC
load FeaturesTSelfattentionResNet18_Self_LBC


%% 
Fuse_Efficient_LBC=[FeaturesTSelfattentionEffcientNet_Self_LBC,FeaturesTSelfattentionEffcientNet_Pool_LBC];
DWT_Fuse_Efficient_LBC=dwt_deepfeatures(Fuse_Efficient_LBC);
DWT_Fuse_Efficient_LBC_Table=table(DWT_Fuse_Efficient_LBC',labels);
%% 
Fuse_Mobile_LBC=[FeaturesTSelfattentionMobileNet_Self_LBC,FeaturesTSelfattentionMobileNet_Pool_LBC];
DWT_Fuse_Mobile_LBC=dwt_deepfeatures(Fuse_Mobile_LBC);
DWT_Fuse_Mobile_LBC_Table=table(DWT_Fuse_Mobile_LBC',labels);

%% 
Fuse_ResNet18_LBC=[FeaturesTSelfattentionResNet18_Self_LBC,FeaturesTSelfattentionResNet18_Pool_LBC];
DWT_Fuse_ResNet18_LBC=dwt_deepfeatures(Fuse_ResNet18_LBC);
DWT_Fuse_ResNet18_LBC_Table=table(DWT_Fuse_ResNet18_LBC',labels);

%%
DWT_ALL_LBC=[DWT_Fuse_Efficient_LBC',DWT_Fuse_Mobile_LBC',DWT_Fuse_ResNet18_LBC'];
DWT_ALL_LBC_Table=table(DWT_ALL_LBC,labels);