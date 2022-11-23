import math
from statistics import geometric_mean

import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.stats
import warnings
import os
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
import time
from utilities import *
import pickle

time_cost = 0
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    meta_data = pd.read_csv('./KONVID_1K_metadata.csv')
    flickr_ids = meta_data.flickr_id
    mos = meta_data.mos.to_numpy()
    data_length = mos.shape[0]

    pca_d = 10
    k = 6

    tem_quality = np.zeros((data_length, 1))
    fused_quality = np.zeros((data_length, 1))
    fused_quality2 = np.zeros((data_length, 1))

    lgn_quality = np.zeros((data_length, 1))
    V1_quality = np.zeros((data_length, 1))
    V1_quality4 = np.zeros((data_length, 1))
    V1_quality8 = np.zeros((data_length, 1))

    niqe_quality = np.zeros((data_length, 1))

    for v in range(data_length):
        time_start = time.time()
        #Read LGN feature of each video
        lgn_feature_mat = scipy.io.loadmat('D:/SourceCode/Features/KoNViD/LGN_Features/LGN_TIP32/' + str(flickr_ids[v]) + '.mp4.mat')
        lgn_feature = lgn_feature_mat['LGN_features_level6']
        lgn_feature = np.asarray(lgn_feature, dtype=np.float)
        lgn_feature = clear_data(lgn_feature)

        # F = open('D:/SourceCode/Features/KoNViD/New_LGN_Features2/features_kon0.25/' + str(flickr_ids[v]) + '.pkl', 'rb')
        # #lgn_feature_mat = scipy.io.loadmat('D:/SourceCode/Features/CVD/CVD_LGN_TIP32/' + flickr_ids[v] + '.mat')
        # lgn_feature = pickle.load(F, encoding='utf-8')#lgn_feature_mat['LGN_features']
        # lgn_feature = np.asarray(lgn_feature, dtype=np.float)
        # lgn_feature = clear_data(lgn_feature)

        V1_feature_mat = scipy.io.loadmat('D:/SourceCode/Features/KoNViD/V1_Features/V1_Gabor0.5/' + str(flickr_ids[v]) + '.mat')
        V1_feature = V1_feature_mat['V1_features']
        V1_feature = np.asarray(V1_feature, dtype=np.float)
        V1_feature = clear_data(V1_feature)

        # nn_feature_mat4 = scipy.io.loadmat('D:/SourceCode/Features/KoNViD/NN_Features/resnet152/resnet152' + str(flickr_ids[v]) + '_pca.mat')
        # nn_feature4 = nn_feature_mat4['features5']
        # nn_feature4 = np.asarray(nn_feature4, dtype=np.float)
        # nn_feature4 = clear_data(nn_feature4)
        #
        # V1_feature_mat4 = scipy.io.loadmat('D:/SourceCode/Features/KoNViD/V1_Features/V1_Gabor0.5/' + str(flickr_ids[v]) + '.mat')
        # V1_feature4 = V1_feature_mat4['V1_features']
        # V1_feature4 = np.asarray(V1_feature4, dtype=np.float)
        # V1_feature4 = clear_data(V1_feature4)

        # V1_feature_mat8 = scipy.io.loadmat('D:/SourceCode/Features/KoNViD/V1_Features/V1_Gabor0.75/' + str(flickr_ids[v]) + '.mat')
        # V1_feature8 = V1_feature_mat8['V1_features']
        # V1_feature8 = np.asarray(V1_feature8, dtype=np.float)
        # V1_feature8 = clear_data(V1_feature8)

        #Read NIQE score of each video
        niqe_score_mat = scipy.io.loadmat('D:/SourceCode/Features/KoNViD/NIQE/'+str(flickr_ids[v])+'.mat')
        niqe_score = niqe_score_mat['features_norm22']

        #Extract pixel features for each video
        # video_name = os.path.join(
        #     'D:\\Dataset\\KoNViD_1k', 'KoNViD_1k_videos\\{}.mp4'.format(int(flickr_ids[v])))
        # img_features = extract_img_features(video_name, 1)

        '''t-SNE'''
        #tsne = TSNE(n_components=2, init='pca')
        #X_tsne = tsne.fit_transform(lgn_feature)

        pca = PCA(n_components=pca_d)
        pca.fit(lgn_feature)
        lgn_PCA = pca.transform(lgn_feature)

        pca = PCA(n_components=pca_d)
        #print(nn_feature4[0])
        pca.fit(V1_feature)
        V1_PCA = pca.transform(V1_feature)
        # lgn_PCA = np.hstack((lgn_PCA, V1_PCA))

        # pca = PCA(n_components=pca_d)
        # pca.fit(V1_feature4)
        # V1_PCA4 = pca.transform(V1_feature4)
        #
        # pca = PCA(n_components=pca_d)
        # pca.fit(V1_feature8)
        # V1_PCA8 = pca.transform(V1_feature8)
        #lgn_curvature = compute_lgn_curvature(lgn_PCA)
        #v1_curvature  = compute_v1_curvature(V1_PCA)
        #lgn_curvature = clear_data(lgn_curvature)
        #v1_curvature = clear_data(v1_curvature)


        lgn_score = compute_lgn_curvature(lgn_PCA)
        v1_score = compute_v1_curvature(V1_PCA)

        # lgn_score = lgn_score[~np.isnan(lgn_score.squeeze())]
        # v1_score = v1_score[~np.isnan(v1_score.squeeze())]
        # lgn_score = lgn_score[~np.isposinf(lgn_score.squeeze())]
        # v1_score = v1_score[~np.isposinf(v1_score.squeeze())]
        # lgn_score = lgn_score[~np.isneginf(lgn_score.squeeze())]
        # v1_score = v1_score[~np.isneginf(v1_score.squeeze())]

        lgn_quality[v] = math.log(np.mean(lgn_score))
        V1_quality[v] = math.log(np.mean(v1_score))
        # V1_quality4[v] = math.log(np.mean(compute_v1_curvature(V1_PCA4)))
        # V1_quality8[v] = math.log(np.mean(compute_v1_curvature(V1_PCA8)))
        # niqe_score = niqe_score[~np.isnan(niqe_score)]
        # niqe_score = niqe_score[~np.isposinf(niqe_score)]
        # niqe_score = niqe_score[~np.isneginf(niqe_score)]

        niqe_quality[v] = np.mean(niqe_score)
        #lgn_quality[v] = np.sqrt(geometric_mean2(lgn_curvature))
        #V1_quality[v] = np.sqrt(geometric_mean2(v1_curvature))
        #tem_quality[v] = math.log(np.mean(linear_pred(V1_PCA, k))) + math.log(np.mean(linear_pred(lgn_PCA, k)))
        #fused_quality[v] = np.mean(np.dot(niqe_score.squeeze()[:len(lgn_PCA)-2], np.log(compute_curvature(lgn_PCA).squeeze())))

        time_end = time.time()
        print('Video {}, overall {} seconds elapsed...'.format(
            v, time_end - time_start))

    temporal_quality = V1_quality + lgn_quality
    data = temporal_quality.squeeze()
    data = data[~np.isnan(data)]
    data = data[~np.isposinf(data)]
    temporal_quality = data[~np.isneginf(data)]

    mu = np.mean(temporal_quality)
    sigma = np.std(temporal_quality)

    mu_niqe = np.mean(niqe_quality)
    sigma_niqe = np.std(niqe_quality)

    niqe_quality = (niqe_quality-mu_niqe)/sigma_niqe*sigma+mu
    print(mu_niqe, sigma_niqe, sigma, mu, len(temporal_quality))

    # temporal_quality = V1_quality + lgn_quality
    # data = temporal_quality.squeeze()
    # data = data[~np.isnan(data)]
    # data = data[~np.isposinf(data)]
    # temporal_quality = data[~np.isneginf(data)]
    #
    # max = np.max(temporal_quality)
    # min = np.min(temporal_quality)
    #
    # max_niqe = np.max(niqe_quality)
    # min_niqe = np.min(niqe_quality)
    #
    # niqe_quality = (niqe_quality-min_niqe)/(max_niqe-min_niqe)*(max-min)+min
    # print(max_niqe, min_niqe, max, min, len(temporal_quality))

    fused_quality = (V1_quality + lgn_quality) * niqe_quality
    fused_quality2 = (V1_quality) * niqe_quality

    # V1_quality, niqe_quality = clear_mos(V1_quality+lgn_quality, niqe_quality)
    # niqe_quality, V1_quality = clear_mos(niqe_quality, V1_quality)
    #
    # plot_scatter(niqe_quality, V1_quality, './image.png', xlabel='NIQE', ylabel='TPQI', haveFit=True)


    # temporal_quality = lgn_quality+V1_quality
    # curvage_mos = temporal_quality
    # fused_mos = mos
    # curvage_mos, fused_mos = clear_mos(temporal_quality, fused_mos)
    # curvage_mos, niqe_mos = clear_mos(temporal_quality, niqe_quality.squeeze())
    # curvage_mos = fit_curve(curvage_mos, fused_mos)
    # niqe_mos = fit_curve(niqe_mos, fused_mos)

    # temporal_quality = lgn_quality+V1_quality
    # curvage_mos = temporal_quality
    # fused_mos = mos
    # curvage_mos, fused_mos = clear_mos(temporal_quality, fused_mos)
    # curvage_mos, niqe_mos = clear_mos(temporal_quality, niqe_quality.squeeze())
    # #curvage_mos = fit_curve(curvage_mos, fused_mos)
    # #niqe_mos = fit_curve(niqe_mos, fused_mos)
    # plot_scatter(curvage_mos, niqe_mos, './image.png', ylabel='Overall', haveFit=True)


    curvage_mos2 = fused_quality
    tem_mos = mos
    curvage_mos2, tem_mos = clear_mos(curvage_mos2, tem_mos)
    curvage_mos2 = fit_curve(curvage_mos2, tem_mos)
    plot_scatter(tem_mos, curvage_mos2, './image.png', ylabel='Overall', haveFit=True)

    curvage_mos2 = (V1_quality + lgn_quality)
    tem_mos = mos
    curvage_mos2, tem_mos = clear_mos(curvage_mos2, tem_mos)
    curvage_mos2 = fit_curve(curvage_mos2, tem_mos)
    plot_scatter(tem_mos, curvage_mos2, './image.png', ylabel='TPQI', haveFit=True)
    #
    curvage_mos = lgn_quality
    fused_mos = mos
    curvage_mos, fused_mos = clear_mos(curvage_mos, fused_mos)
    curvage_mos = fit_curve(curvage_mos, fused_mos)
    plot_scatter(fused_mos, curvage_mos, './image.png', ylabel='Curvatures Correlation', haveFit=True)
    #
    curvage_mos = V1_quality
    tem_mos = mos
    curvage_mos, tem_mos = clear_mos(curvage_mos, tem_mos)
    curvage_mos = fit_curve(curvage_mos, tem_mos)
    plot_scatter(tem_mos, curvage_mos, './image.png', ylabel='NIQE', haveFit=True)

    #plot_scatter(V1_quality.squeeze(), lgn_quality.squeeze(), './image.png', xlabel = 'Spatial Quality Scores', ylabel='Temporal Quality Scores', haveFit=True)
