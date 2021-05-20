# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Dec. 2019

Warning: These functions are not part of the MMLL library, 
they are only intended for demostration purposes. Use at your own risk!


'''
import json
import numpy as np
import time
from sklearn.metrics import roc_curve, auc
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib
#matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sn
import os, sys

def display(message, logger, verbose, uselog=True):
    if verbose:
        print(message)
    if uselog:
        try:
            logger.info(message)
        except:
            pass
         
def create_folders(path):
    # Create the directories for storing relevant outputs if they do not exist
    if not os.path.exists(path + "logs/"):
        os.makedirs(path + "logs/") # Create directory for the logs
    if not os.path.exists(path + "figures/"):
        os.makedirs(path + "figures/") # Create directory for the figures
    if not os.path.exists(path + "models/"):
        os.makedirs(path + "models/") # Create directory for the models
   
def format_fn(tick_val, tick_pos):
    if int(tick_val) in xs:
        return labels[int(tick_val)]
    else:
        return ''

def eval_regression(pom, model_type, dataset_name, Xval_b, yval, Xtst_b, ytst, preds_val, preds_tst, model, logger, verbose):
    Xval_b = np.array(Xval_b).astype(float)
    yval = np.array(yval).astype(float)
    Xtst_b = np.array(Xtst_b).astype(float)
    ytst = np.array(ytst).astype(float)

    NMSE_val = np.linalg.norm(preds_val.ravel() - yval.ravel()) ** 2 / np.linalg.norm(yval) ** 2
    NMSE_tst = np.linalg.norm(preds_tst.ravel() - ytst.ravel()) ** 2 / np.linalg.norm(yval) ** 2
    display('\n===================================================================', logger, verbose)
    display('NMSE on validation set = %s' % str(NMSE_val)[0: 6], logger, verbose)
    display('NMSE on test set = %s' % str(NMSE_tst)[0: 6], logger, verbose)
    display('===================================================================\n', logger, verbose)

    # plotting 1D examples
    if dataset_name in ['lin1D', 'sinc1D']:
        fig = plt.figure(figsize=(10, 8))

        if dataset_name in ['lin1D'] and model_type in ['RR', 'LR']:
            Xvalplot = Xval_b[:, 1].ravel()
            Xtstplot = Xtst_b[:, 1].ravel()

        if dataset_name in ['lin1D'] and model_type in ['KR_pm', 'KR']:
            Xvalplot = Xval_b[:, 0].ravel()
            Xtstplot = Xtst_b[:, 0].ravel()

        if dataset_name in ['sinc1D'] and model_type in ['KR_pm', 'KR']:
            Xvalplot = Xval_b[:, 0].ravel()
            Xtstplot = Xtst_b[:, 0].ravel()
            
        display_text_on_legend = 'Targets'
        index = np.argsort(Xvalplot)
        plt.plot(Xvalplot[index], yval.ravel()[index], 'b.', linewidth=3.0, markersize=3, label=display_text_on_legend)

        if dataset_name in ['sinc1D'] and model_type in ['KR_pm', 'KR']:
            plt.plot(model.C.ravel(), (0 * model.C).ravel(), 'co', linewidth=3.0, markersize=3)

        display_text_on_legend = 'Predictions'
        plt.plot(Xvalplot[index], preds_val.ravel()[index], 'r', linewidth=3.0, markersize=3, label=display_text_on_legend)

        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc="best")
        plt.title(model_type + ' estimation (validation set)')
        plt.grid(True)
        #plt.show()
        output_filename = './results/figures/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_val.png'
        plt.savefig(output_filename)
        display('===================================================================', logger, verbose)
        display('Master_' + model_type + ':saved figure in %s' % output_filename, logger, verbose)

        fig = plt.figure(figsize=(10, 8))
        display_text_on_legend = 'Targets'
          
        index = np.argsort(Xtstplot)
        plt.plot(Xtstplot[index], ytst.ravel()[index], 'b.', linewidth=3.0, markersize=3, label=display_text_on_legend)       
        display_text_on_legend = 'Predictions'

        if dataset_name in ['sinc1D'] and model_type in ['KR_pm', 'KR']:
            plt.plot(model.C.ravel(), (0 * model.C).ravel(), 'co', linewidth=3.0, markersize=3)

        plt.plot(Xtstplot[index], preds_tst.ravel()[index], 'r', linewidth=3.0, markersize=3, label=display_text_on_legend)
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc="best")
        plt.title(model_type + ' estimation (test set)')
        plt.grid(True)
        #plt.show()
        output_filename = './results/figures/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_tst.png'
        plt.savefig(output_filename)
        display('Master_' + model_type + ':saved figure in %s' % output_filename, logger, verbose)
        display('===================================================================\n', logger, verbose)

    if dataset_name in ['redwine', 'ypmsd', 'lin1D']:
        # We plot the sorted targets vs. the predicted values
        fig = plt.figure(figsize=(10, 8))
        index = np.argsort(preds_val.ravel())
        display_text_on_legend = 'Targets'
        plt.plot(yval.ravel()[index], 'b', linewidth=3.0, markersize=3, label=display_text_on_legend)
        display_text_on_legend = 'Predictions'
        plt.plot(preds_val.ravel()[index], 'r.', linewidth=3.0, markersize=3, label=display_text_on_legend)
        #plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc="best")
        plt.title('Sorted Predictions and Targets (validation set)')
        plt.grid(True)
        #plt.show()
        output_filename = './results/figures/POM' + str(pom) + '_' + model_type + '_preds_' + dataset_name + '_val.png'
        plt.savefig(output_filename)
        display('===================================================================', logger, verbose)
        display('Master_' + model_type + ':saved figure in %s' % output_filename, logger, verbose)

        fig = plt.figure(figsize=(10, 8))
        index = np.argsort(preds_tst.ravel())
        display_text_on_legend = 'Targets'
        plt.plot(ytst.ravel()[index], 'b', linewidth=3.0, markersize=3, label=display_text_on_legend)
        display_text_on_legend = 'Predictions'
        plt.plot(preds_tst.ravel()[index], 'r.', linewidth=3.0, markersize=3, label=display_text_on_legend)
        #plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc="best")
        plt.title('Sorted Predictions and Targets (test set)')
        plt.grid(True)
        #plt.show()
        output_filename = './results/figures/POM' + str(pom) + '_' + model_type + '_preds_' + dataset_name + '_tst.png'
        plt.savefig(output_filename)
        display('Master_' + model_type + ':saved figure in %s' % output_filename, logger, verbose)
        display('===================================================================\n', logger, verbose)
    return

def eval_classification(pom, model_type, dataset_name, Xval_b, yval, Xtst_b, ytst, preds_val, preds_tst, logger, verbose, model, output_folder='./results/figures/'):
    Xval_b = np.array(Xval_b).astype(float)
    yval = np.array(yval).astype(float)
    Xtst_b = np.array(Xtst_b).astype(float)
    ytst = np.array(ytst).astype(float)
    roc_auc_val = None
    if preds_val is not None:
        fpr_val, tpr_val, thresholds_val = roc_curve(list(yval), preds_val)
        roc_auc_val = auc(fpr_val, tpr_val)

    fpr_tst, tpr_tst, thresholds_tst = roc_curve(list(ytst), preds_tst)
    roc_auc_tst = auc(fpr_tst, tpr_tst)

    fig = plt.figure(figsize=(10, 8))
    MS = 5
    if preds_val is not None:
        display_text_on_legend = 'Validation set, AUC = %s' % str(roc_auc_val)[0:5]
        plt.plot(fpr_val, tpr_val, 'b', linewidth=6.0, markersize=3, label=display_text_on_legend)
    display_text_on_legend = 'Test set, AUC = %s' % str(roc_auc_tst)[0:5]
    plt.plot(fpr_tst, tpr_tst, 'g', linewidth=6.0, markersize=3, label=display_text_on_legend)

    plt.axis([0, 1.0, 0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curves for ' + dataset_name)
    plt.legend(loc="best")
    plt.grid(True)
    #plt.show()
    output_filename = output_folder + 'POM' + str(pom) + '_ROC_' + model_type + '_' + dataset_name + '.png'
    plt.savefig(output_filename)
    
    display('===================================================================', logger, verbose, uselog=False)
    if preds_val is not None:
        display('AUC on validation set = %f' % roc_auc_val, logger, verbose)
    display('AUC on test set = %f' % roc_auc_tst, logger, verbose)
    display('Master: saved figure at %s' % output_filename, logger, verbose)

    if dataset_name in ['synth2D-class']:
        # drawing contour
        delta = 0.025
        x = np.arange(-1.5, 1.5, delta)
        y = np.arange(-1.5, 1.5, delta)
        X, Y = np.meshgrid(x, y)

        XX = []
        M = x.shape[0]
        N = x.shape[0]
        for m in range(M):
            for n in range(N):
                XX.append((X[m, n], Y[m, n]))

        XX = np.array(XX)
        preds = model.predict(XX)

        Z = np.zeros((M, N))
        count = 0
        for m in range(M):
            for n in range(N):
                Z[m, n] = preds[count][0]
                count += 1               

        fig = plt.figure(figsize=(10, 8))
        which = (yval == 0).ravel()
        plt.plot(Xval_b[which, 0], Xval_b[which,1], 'b.', linewidth=3.0, markersize=MS)
        which = (yval == 1).ravel()
        plt.plot(Xval_b[which, 0], Xval_b[which,1], 'r.', linewidth=3.0, markersize=MS)
        plt.plot(model.C[:, 0], model.C[:, 1], 'cx')
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.title('Classification of validation data')
        plt.grid(True)

        plt.contour(X, Y, Z, levels=[-1, 0, 1], colors=['g', 'k', 'g'], linestyles=['dashed', 'solid', 'dashed'])
        #fig, ax = plt.subplots()
        #CS = ax.contour(X, Y, Z)
        #ax.clabel(CS, inline=1, fontsize=10)

        output_filename = output_folder + 'POM' + str(pom) + '_contour_' + model_type + '_' + dataset_name + '_val.png'
        plt.savefig(output_filename)
        #plt.show()
        #display('Master: saved figure at %s' % output_filename, logger, verbose)

        fig = plt.figure(figsize=(10, 8))
        which = (ytst == 0).ravel()
        plt.plot(Xtst_b[which, 0], Xtst_b[which,1], 'b.', linewidth=3.0, markersize=MS)
        which = (ytst == 1).ravel()
        plt.plot(Xtst_b[which, 0], Xtst_b[which,1], 'r.', linewidth=3.0, markersize=MS)
        plt.plot(model.C[:, 0], model.C[:, 1], 'cx')
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.title('Classification of test data')
        plt.grid(True)
        plt.contour(X, Y, Z, levels=[-1, 0, 1], colors=['g', 'k', 'g'], linestyles=['dashed', 'solid', 'dashed'])
        #fig, ax = plt.subplots()
        #CS = ax.contour(X, Y, Z)
        #ax.clabel(CS, inline=1, fontsize=10)

        output_filename = output_folder + 'POM' + str(pom) + '_contour_' + model_type + '_' + dataset_name + '_tst.png'
        plt.savefig(output_filename)
        #plt.show()
        display('Master: saved figure at %s' % output_filename, logger, verbose)

    display('===================================================================\n', logger, verbose, uselog=False)
    return roc_auc_val, roc_auc_tst


def eval_clustering(pom, model_type, dataset_name, Xtst_b, c, logger, verbose):
    Xtst_b = np.array(Xtst_b).astype(float)
    # Plotting 2D
    display('\n===================================================================', logger, verbose)

    if dataset_name in ['synth2D', 'synth2Db', 'synth2Dc', 'joensuu', 'synth2D-class']:
        fig = plt.figure(figsize=(10, 8))
        plt.plot(Xtst_b[:, 0], Xtst_b[:, 1], 'b.', linewidth=6.0, markersize=3)
        plt.plot(c[:, 0], c[:, 1], 'r*', linewidth=6.0, markersize=10)
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(model_type + ' clustering')
        plt.grid(True)
        #plt.show()
        output_filename = './results/figures/POM' + str(pom) + '_clusters_' + model_type + '_' + dataset_name + '.png'
        plt.savefig(output_filename)
        display('Master_' + model_type + ':saved figure in %s' % output_filename, logger, verbose)

    # Plotting 1D
    if dataset_name in ['lin1D', 'sinc1D']:
        fig = plt.figure(figsize=(10, 8))
        plt.plot(Xtst_b[:, 0], 0 * Xtst_b[:, 0], 'b.', linewidth=6.0, markersize=3)
        plt.plot(c[:, 0], 0 * c[:, 0], 'r*', linewidth=6.0, markersize=10)
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(model_type + ' clustering')
        plt.grid(True)
        #plt.show()
        output_filename = './results/figures/POM' + str(pom) + '_clusters_' + model_type + '_' + dataset_name + '.png'
        plt.savefig(output_filename)
        display('Master_' + model_type + ':saved figure in %s' % output_filename, logger, verbose)

    # Plotting images
    if dataset_name in ['mnist', 'mnist-small', 'fashion', 'M-fashion', 'M-fashion-small']:
        NC = c.shape[0]

        fig, axs = plt.subplots(nrows=6, ncols=7, figsize=(10, 8),
                                subplot_kw={'xticks': [], 'yticks': []})
        axes = list(axs.ravel())
        for kc in range(0, NC):
            digit = c[kc, :].reshape((28, 28))
            ax = axes[kc]
            ax.imshow(digit, cmap='viridis')
        plt.tight_layout()
        #plt.show()
        output_filename = './results/figures/POM' + str(pom) + '_clusters_' + model_type + '_' + dataset_name + '.png'
        plt.savefig(output_filename)
        display('Master_' + model_type + ':saved figure in %s' % output_filename, logger, verbose)

    # saving centroids in all cases
    #output_filename_c = './results/models/POM' + str(pom) + '_C_' + model_type + '_' + dataset_name + '.pkl'
    #with open(output_filename_c, 'wb') as f:
    #    pickle.dump(c, f)
    #display('Master_' + model_type + ':saved centroids in %s' % output_filename_c, logger, verbose)
    display('===================================================================\n', logger, verbose)

    return

def eval_xcorr(pom, model_type, dataset_name, list_self_corrs, list_cross_corrs, N, logger, verbose):
    Ny = len(list_self_corrs)
    Nshow = np.min((Ny, N))
    y = list_self_corrs[0: Nshow]
    values = [yy[2] for yy in y]
    labels = ['$x_{%s}x_{%s}$' % (yy[0], yy[1]) for yy in y]
    x = range(Nshow)

    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 8))
    plt.bar(x, values)
    plt.title('Largest normalized correlation values among inputs')
    plt.ylabel('Correlation')
    plt.xticks(x, labels)
    #locs, labels = plt.xticks()

    plt.grid(True)
    output_filename = './results/figures/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_selfcorr.png'
    plt.savefig(output_filename)
    display('\n===================================================================', logger, verbose)
    display('Master_' + model_type + ':saved figure in %s' % output_filename, logger, verbose)
    
    Ny = len(list_cross_corrs)
    Nshow = np.min((Ny, N))
    y = list_cross_corrs[0: Nshow]
    values = [yy[1] for yy in y]
    labels = ['$x_{%s}y$' % yy[0] for yy in y]

    x = range(Nshow)
    fig = plt.figure(figsize=(10, 8))
    plt.bar(x, values)
    plt.title('Largest normalized correlation values between input-output')
    plt.ylabel('Correlation')
    plt.xticks(x, labels)
    plt.grid(True)
    output_filename = './results/figures/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_crosscorr.png'
    plt.savefig(output_filename)
    display('Master_' + model_type + ':saved figure in %s' % output_filename, logger, verbose)
    display('===================================================================\n', logger, verbose)
    return

def eval_multiclass_classification(pom, model_type, dataset_name, Xval_b, yval, Xtst_b, ytst, logger, verbose, mn, classes, preds_val_dict, preds_tst_dict, o_val, o_tst, figures_folder='./'):
    try:
        Xval_b = np.array(Xval_b).astype(float)
        yval = np.array(yval)
        Xtst_b = np.array(Xtst_b).astype(float)
        ytst = np.array(ytst)
        colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y', 'k--', 'r--', 'g--', 'b--', 'm--']
        fig = plt.figure(figsize=(10, 8))
        for idx, cla in enumerate(classes):
            #yval_ = np.array([str(int(y)) == cla for y in list(yval)]).astype(float)
            yval_ = np.array([y == cla for y in list(yval)]).astype(float)
            fpr_val, tpr_val, thresholds_val = roc_curve(yval_, preds_val_dict[cla])
            roc_auc_val = auc(fpr_val, tpr_val)
            if str(roc_auc_val) != 'nan':
                display_text_on_legend = 'Class %s, AUC = %s' % (cla, str(roc_auc_val)[0:7])
            else: 
                display_text_on_legend = 'Class %s, AUC = %s' % (cla, 'N.A.')
            plt.plot(fpr_val, tpr_val, colors[idx], linewidth=3.0, markersize=3, label=display_text_on_legend)

        plt.axis([0, 1.0, 0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curves for ' + dataset_name + ', validation set.')
        plt.legend(loc="best")
        plt.grid(True)
        #plt.show()
        output_filename = figures_folder + 'POM_' + str(pom) + '_ROC_' + model_type + '_' + dataset_name + '_val.png'
        plt.savefig(output_filename)
        display('Master_' + model_type + ':saved figure at %s' % output_filename, logger, verbose)

        fig = plt.figure(figsize=(10, 8))
        for idx, cla in enumerate(classes):
            #ytst_ = np.array([str(int(y)) == cla for y in list(ytst)]).astype(float)
            ytst_ = np.array([y == cla for y in list(ytst)]).astype(float)
            fpr_tst, tpr_tst, thresholds_tst = roc_curve(ytst_, preds_tst_dict[cla])
            roc_auc_tst = auc(fpr_tst, tpr_tst)
            if str(roc_auc_tst) != 'nan':
                display_text_on_legend = 'Class %s, AUC = %s' % (cla, str(roc_auc_tst)[0:7])
            else: 
                display_text_on_legend = 'Class %s, AUC = %s' % (cla, 'N.A.')
            plt.plot(fpr_tst, tpr_tst, colors[idx], linewidth=3.0, markersize=3, label=display_text_on_legend)
            print(cla, roc_auc_tst)

        plt.axis([0, 1.0, 0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curves for ' + dataset_name + ', test set.')
        plt.legend(loc="best")
        plt.grid(True)
        #plt.show()
        output_filename = figures_folder + 'POM_' + str(pom) + '_ROC_' + model_type + '_' + dataset_name + '_tst.png'
        plt.savefig(output_filename)
        display('Master_' + model_type + ':saved figure at %s' % output_filename, logger, verbose)

        # True, predicted
        yval_ = list(yval.ravel())
        #yval_ = [y for y in yval_]
        CM_val = confusion_matrix(yval_, o_val)

        ytst_ = list(ytst.ravel())
        #ytst_ = [str(int(y)) for y in ytst_]
        #ytst_ = [y for y in ytst_]
        CM_tst = confusion_matrix(ytst_, o_tst)

        df_cm = pd.DataFrame(CM_val, index=classes, columns=classes)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.title('Confussion matrix for ' + dataset_name + ', validation set.')
        output_filename = figures_folder + 'POM_' + str(pom) + '_CM_' + model_type + '_' + dataset_name + '_val.png'
        plt.savefig(output_filename)
        display('Master_' + model_type + ':saved figure at %s' % output_filename, logger, verbose)

        df_cm = pd.DataFrame(CM_tst, index=classes, columns=classes)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.title('Confussion matrix for ' + dataset_name + ', test set.')
        output_filename = figures_folder + 'POM_' + str(pom) + '_CM_' + model_type + '_' + dataset_name + '_tst.png'
        plt.savefig(output_filename)
        display('Master_' + model_type + ':saved figure at %s' % output_filename, logger, verbose)
        display('===================================================================\n', logger, verbose)
    except Exception as err:
        print('STOP AT eval_multiclass_classification')
        import code
        code.interact(local=locals())
    return


def Kmeans_plot(X, preds, title, output_filename, logger, verbose):
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit(X).transform(X)
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=preds)
        plt.xlabel('PCA component 1')
        plt.ylabel('PCA component 2')
        plt.title(title)
        plt.grid(True)
        output_filename = './results/figures/' + output_filename
        plt.savefig(output_filename)
    except:
        display('Model not correctly trained, not drawing', logger, verbose)



def plot_cm_seaborn(preds, y, classes, title, output_filename, logger, verbose, normalize=False, cmap=plt.cm.GnBu):
    cnf_matrix = confusion_matrix(y, preds)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'        
        
    df = pd.DataFrame(cnf_matrix, columns=classes, index=classes)
    plt.figure(figsize = (10, 8))
    ax = sn.heatmap(df, annot=True, cmap=cmap, linewidths=.5, cbar=False, fmt=fmt)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    output_filename = './results/figures/' + output_filename
    plt.savefig(output_filename)


