clear all
clc

load SelfAttentionCervical_Sky_EfficientnetB0_channels1280

inputSize = net.Layers(1).InputSize(1:2);
%%
% img1 = imread("HSIL_1 (1).jpg");
% img2 = imread("LSIL_3 (3).jpg");
% img3 = imread("NL_1_ (6).jpg");
% img4 = imread("scc_1 (1).jpg");
img5 = imread("Superficial-Intermediate.bmp");
img6 = imread('Parabasal.bmp');
img7 = imread("Metaplastic.bmp");
img8= imread('Koilocytotic.bmp');
img9= imread('Dyskeratotic.bmp');


img1 = imresize(img1,inputSize);
img2 = imresize(img2,inputSize);
img3 = imresize(img3,inputSize);
img4 = imresize(img4,inputSize);

% img4 = imresize(img4,inputSize);
% img5 = imresize(img5,inputSize);
% img6 = imresize(img6,inputSize);
% img7 = imresize(img7,inputSize);

[label1,score1] = classify(net,img1);
%score1 = predict(net,(img1));

%[~, channel1] = max(score1);
[label2,score2] = classify(net,img2);
[label3,score3] = classify(net,img3);
[label4,score4] = classify(net,img4);
%[label5,score5] = classify(net,img5);
% [label6,score] = classify(net,img6);
% [label7,score] = classify(net,img7);

scoreMap1 = gradCAM(net,img1,label1);
scoreMap2 = gradCAM(net,img2,label2);
scoreMap3 = gradCAM(net,img3,label3);
scoreMap4 = gradCAM(net,img4,label4);
%scoreMap5 = gradCAM(net,img5,label5);
% scoreMap6 = imageLIME(net,img6,label6);
% scoreMap7 = imageLIME(net,img7,label7);
% scoreMap1 = imageLIME(net,img1,label1);
% scoreMap2 = imageLIME(net,img2,label2);
% scoreMap3 = imageLIME(net,img3,label3);
% scoreMap4 = imageLIME(net,img4,label4);
figure
imshow(img1)
 hold on
imagesc(scoreMap1,'AlphaData',0.5)
colormap jet


figure
imshow(img2)
hold on
imagesc(scoreMap2,'AlphaData',0.5)
colormap jet
figure
imshow(img3)
hold on
imagesc(scoreMap3,'AlphaData',0.5)
colormap jet

figure
imshow(img4)
hold on
imagesc(scoreMap4,'AlphaData',0.5)
colormap jet

% figure
% imshow(img5)
% hold on
% imagesc(scoreMap5,'AlphaData',0.5)
% colormap jet
%%

clear all
clc

load SelfAttentionCervical_LBC_ResNet18_channels512
%load Cervical_LBC_EfficientnetB0
%load SelfAttentionCervical_LBC_ResNet18_channels512_Selflayer

inputSize = net.Layers(1).InputSize(1:2);

% img1 = imread("Dyskeratotic.bmp");
% img2 = imread("Koilocytotic.bmp");
% img3 = imread("Metaplastic.bmp");
% img4 = imread("Parabasal.bmp");
% img5 = imread("Superficial-Intermediate.bmp");
% img6 = imread("nv.jpg");
% img7 = imread("vasc.jpg");
img1 = imread("HSIL_1 (1).jpg");
img2 = imread("LSIL_3 (3).jpg");
img3 = imread("NL_1_ (6).jpg");
img4 = imread("scc_1 (1).jpg");


img1 = imresize(img1,inputSize);
img2 = imresize(img2,inputSize);
img3 = imresize(img3,inputSize);
img4 = imresize(img4,inputSize);
%img5 = imresize(img5,inputSize);
% img4 = imresize(img4,inputSize);
% img5 = imresize(img5,inputSize);
% img6 = imresize(img6,inputSize);
% img7 = imresize(img7,inputSize);

[label1,score1] = classify(net,img1);
[label2,score2] = classify(net,img2);
[label3,score3] = classify(net,img3);
[label4,score4] = classify(net,img4);
%[label5,score5] = classify(net,img5);
% [label6,score] = classify(net,img6);
% [label7,score] = classify(net,img7);
[~, channel1] = max(score1);
[~, channel2] = max(score2);
[~, channel3] = max(score3);
[~, channel4] = max(score4);
%[~, channel5] = max(score5);

scoreMap1 = gradCAM(net,img1,label1);
scoreMap2 = gradCAM(net,img2,label2);
scoreMap3 = gradCAM(net,img3,label3);
scoreMap4 = gradCAM(net,img4,label4);
% scoreMap5 = gradCAM(net,img5,label5);
% scoreMap6 = imageLIME(net,img6,label6);
% scoreMap7 = imageLIME(net,img7,label7);
% scoreMap1 = imageLIME(net,img1,"High squamous intra-epithelial lesion");
% scoreMap2 = imageLIME(net,img2,"High squamous intra-epithelial lesion");
% scoreMap3 = imageLIME(net,img3,"High squamous intra-epithelial lesion");
% scoreMap4 = imageLIME(net,img4,"High squamous intra-epithelial lesion");
%scoreMap5 = imageLIME(net,img5,'Dyskeratotic');
figure
imshow(img1)
 hold on
imagesc(scoreMap1,'AlphaData',0.5)
colormap jet


figure
imshow(img2)
hold on
imagesc(scoreMap2,'AlphaData',0.5)
colormap jet
figure
imshow(img3)
hold on
imagesc(scoreMap3,'AlphaData',0.5)
colormap jet

figure
imshow(img4)
hold on
imagesc(scoreMap4,'AlphaData',0.5)
colormap jet

% figure
% imshow(img5)
% hold on
% imagesc(scoreMap5,'AlphaData',0.5)
% colormap jet