
from matplotlib import pyplot as plt
import numpy as np
import cmath
from sklearn.decomposition import PCA
from scipy.fftpack import fft, ifft
import pickle
from multiprocessing import Pool, cpu_count, Value, Lock, Manager
import os
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count, Value, Lock, Manager
from tqdm import tqdm
from sklearn.decomposition import FastICA

class TG(object):

    
    def __init__(self,data,folder_name,folder_path,poses,a=0,width=280,**kwargs):
        self.data=data
        self.a = a
        self.folder_name=folder_name
        self.folder_path=folder_path
        self.width = width
        self.poses=poses
        self.prepare_data()
        self.TTSR=0
        self.mean_first_derivative=0
        self.mean_second_derivative=0 
  
        
     ###Precprocessing
    #StrtEndMainMethod      
    def prepare_data(self):
        H, W = self.data.shape[:2]
        x0 = W // 2 
        y0 = H // 2
        x=50
        #background determination
        if np.std(self.data[x0,y0,10:45])*4>np.std(self.data[x0,y0,50:90]):
            bck=95
            self.freq=100
        else:
            self.freq=50
            bck=45
        global_max=np.argmax(np.mean(self.data[x0-x:x0+x,y0-x:y0+x,:],axis=(0,1)))+self.a
        sy,sx=self.poses
        T_zero=np.mean(self.data[:,:,:bck],axis=2)
        TT=self.data[:,:,:]-T_zero[:,:,None]
        TT[TT <= 0] = np.nan
        TT=self.replace_inf_nan_with_avg(TT)
        width=self.width
        TT_new = TT[:, :, global_max : global_max + width]
        #print(sy, sx)
        #print(TT_new.shape)
        TT_new2=TT_new[sy, sx, :]
        #print(TT_new2.shape)
        self.TT=TT_new2
        # Option 1: Per-pixel normalization over time (RECOMMENDED)
        # mean = self.TT.mean(axis=2, keepdims=True)  # shape: (h, w, 1)
        # std = self.TT.std(axis=2, keepdims=True)    # shape: (h, w, 1)
        # TT_normalized = (self.TT - mean) / (std + 1e-8)  # still (h, w, n_frames)

        # Option 2: Global normalization (simpler)
        TT_normalized = (self.TT - self.TT.mean()) / self.TT.std()
        self.TT=TT_normalized
        return
    def replace_inf_nan_with_avg(self,TT):
        arr=TT
        nan_inf_mask = np.isnan(arr) | np.isinf(arr)
        avg_values = np.nanmean(arr, axis=(0, 1), keepdims=True)
        avg_values = np.broadcast_to(avg_values, arr.shape)
        arr[nan_inf_mask] = avg_values[nan_inf_mask]
        return arr
    #NanRemoveMainMethod
    def replace_inf_nan_Tlog(self):
        Tlog=np.log(self.TT[:,:,:])
        self.Tlog=self.replace_inf_nan_with_avg(Tlog)
        return
    
    
    ##TSR##
    @staticmethod
    def fit_func(x, *args):
        f=args[0]
        for i in range(1,len(args)):
            f+=args[i]*x**(i)
        return f
    @staticmethod
    def fit_curve(params):
        i, j, tlg, T1lg, n, counter, lock = params
        pini=np.zeros(n)
        pini[0]=1
        params = curve_fit(TG.fit_func, tlg, T1lg[i,j,:], p0=pini)
    
        with lock:
            counter.value += 1
            if counter.value % (len(T1lg[0])*100) == 0:
                print(f"Processed row {counter.value // len(T1lg[0])}/{len(T1lg)}")
    
        return (i, j, params[0])
    #TSRMainMethod
    def TSR(self,n=6,**kwargs):
        self.replace_inf_nan_Tlog() 
        T1lg=np.copy(self.Tlog)
        t=np.linspace(0.1,len(T1lg[1,1,:])-1,len(T1lg[1,1,:]))

        tlg=np.log(t)
        TTSR=np.zeros((len(T1lg[:,1,1]),len(T1lg[1,:,1]),n))

        manager = Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        params_list = [(i, j, tlg, T1lg, n, counter, lock) for i in range(len(TTSR[:,1,1])) for j in range(len(TTSR[1,:,1]))]

        with Pool(cpu_count()) as pool:
            results = pool.map(TG.fit_curve, params_list)
    
        for i, j, result in results:
            TTSR[i,j,:] = result
        self.TTSR=TTSR  
        return 
    ##PPT##
    def FFTPPT(self,**kwargs):
        T=self.TT
        freq = self.freq
        TPPT=np.zeros_like(T[:,:,:],dtype=np.complex128)
        TPPT = fft(T, axis=2)
        N = len(T[1,1,:])
        n = np.arange(N)
        Tau = N/freq
        freqScale = n/Tau
        Phase=np.zeros_like(TPPT[:,:,:],dtype=np.float64)
        Amp=np.zeros_like(TPPT[:,:,:],dtype=np.float64)
        for k in range(0, len(TPPT[1,1,:])):
            Phase[:,:,k]=np.arctan2(TPPT[:,:,k].imag,TPPT[:,:,k].real)
            Amp[:,:,k]=np.abs(TPPT[:,:,k])
            muA = np.mean(Amp[:, :, k])
            stdA = np.std(Amp[:, :, k])
            Amp[:, :, k] = (Amp[:, :, k] - muA) / (stdA+1e-8)

            muP = np.mean(Phase[:, :, k])
            stdP = np.std(Phase[:, :, k])
            Phase[:, :, k] = (Phase[:, :, k] - muP) / (stdP+1e-8)

        self.Phase=Phase
        self.Amp=Amp
        self.freqScale=freqScale
        return 
    ##PCA##
    def PCAMtr(self,**kwargs):
        m = kwargs.get('mPCA', 6)
        T=self.TT
        pca = PCA(m)
        MPCA=T.reshape(len(T[:,1,1])*len(T[1,:,1]),len(T[1,1,:]))
        converted_data = pca.fit_transform(MPCA)
        converted_data=converted_data.reshape(len(T[:,1,1]),len(T[1,:,1]),m)
        self.converted_data=converted_data
        return 
    

    def ICAMtr(self, **kwargs):
        m = kwargs.get('mICA', 6)
        T = self.TT

        ica = FastICA(n_components=m, random_state=0, max_iter=1000)

        # reshape: (H*W, T)
        MICA = T.reshape(len(T[:, 1, 1]) * len(T[1, :, 1]),
                        len(T[1, 1, :]))

        ICA_data = ica.fit_transform(MICA)

        # reshape back to (H, W, m)
        ICA_data = ICA_data.reshape(len(T[:, 1, 1]),
                                                len(T[1, :, 1]),
                                                m)

        self.ICA_data = ICA_data
        self.ica_model = ica
        return
    
    def CalcMeth(self,**kwargs):
        self.FFTPPT(self,**kwargs)
        self.PCAMtr(self,**kwargs)
        return 
    def find_contrast_limits(self,image, nbins=50):
        """Find vmin/vmax based on histogram peak FWHM."""
        counts, edges = np.histogram(image.ravel(), bins=nbins)
        centers = (edges[:-1] + edges[1:]) / 2
        
        # Find highest peak
        peak_idx = np.argmax(counts)
        peak_val = counts[peak_idx]
        peak_pos = centers[peak_idx]
        
        # Find FWHM
        half_max = peak_val / 2
        # Find indices where counts cross half_max
        above_half = counts > half_max
        
        # Find left and right edges of FWHM
        left_idx = peak_idx
        right_idx = peak_idx
        
        # Search left
        for i in range(peak_idx, -1, -1):
            if counts[i] < half_max:
                left_idx = i
                break
        
        # Search right  
        for i in range(peak_idx, len(counts)):
            if counts[i] < half_max:
                right_idx = i
                break
        
        # Calculate FWHM in terms of data values
        fwhm = centers[right_idx] - centers[left_idx]
        
        vmin = peak_pos - 0.58 * fwhm
        vmax = peak_pos + 0.75 * fwhm
        
        return vmin, vmax

    def plot_image_with_histogram(self,image,name, nbins=50):
        """Plot image and histogram with contrast adjusted by FWHM."""
        vmin, vmax = self.find_contrast_limits(image, nbins)
        save_dir = os.path.join(self.folder_path, "preliminary_check_plots",self.folder_name,"global_max_"+"a="+str(self.a)+"_width="+str(self.width))#,"a="+str(self.a)+"_b="+str(self.b)
        os.makedirs(save_dir, exist_ok=True)  # Create folder if not existing
        save_path = os.path.join(save_dir, f"{name}.png")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(self.folder_name+" "+name, fontsize=14)
        # Plot image with adjusted contrast
        im = ax1.imshow(image, vmin=vmin, vmax=vmax)
        ax1.set_title(f'Image (vmin={vmin:.2f}, vmax={vmax:.2f})')
        plt.colorbar(im, ax=ax1)
        
        # Plot histogram
        counts, edges, _ = ax2.hist(image.ravel(), bins=nbins, color='blue', alpha=0.7)
        ax2.axvline(vmin, color='red', linestyle='--', label=f'vmin={vmin:.2f}')
        ax2.axvline(vmax, color='red', linestyle='--', label=f'vmax={vmax:.2f}')
        ax2.set_xlabel('Pixel Value')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Histogram (nbins={nbins})')
        ax2.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(save_path, dpi=200)
        #print(f"fig saved in {save_path}")
        plt.close(fig)
    def SavePlots_PCA_Phase(self): 
        for i in range(self.converted_data.shape[2]-1):
            self.plot_image_with_histogram(self.converted_data[:,:, i+1], name=f'PCA_Component_{i+2}')#self.poses[0], self.poses[1]
        for i in range(7):
            self.plot_image_with_histogram(self.Phase[:,:, i+1], name=f'FFT_Phase_Freq_{i+2}')#self.poses[0], self.poses[1]
        return
    def SavePlots_PPT(self): 
        for i in range(7):
            self.plot_image_with_histogram(self.Phase[:,:, i+1], name=f'FFT_Phase_Freq_{i+2}')#self.poses[0], self.poses[1]
        for i in range(7):
            self.plot_image_with_histogram(self.Amp[:,:, i], name=f'FFT_Amp_Freq_{i+1}')#self.poses[0], self.poses[1]
        return
    def SavePlots_PCA(self): 
        for i in range(self.converted_data.shape[2]-1):
            self.plot_image_with_histogram(self.converted_data[:,:, i+1], name=f'PCA_Component_{i+2}')#self.poses[0], self.poses[1]
        return
    def SavePlots_ICA(self): 
        for i in range(self.ICA_data.shape[2]):
            self.plot_image_with_histogram(self.ICA_data[:,:, i], name=f'ICA_Component_{i}')
        return
    def SavePlots_TSR(self): 
        for i in range(self.TTSR.shape[2]):
            self.plot_image_with_histogram(self.TTSR[:,:, i], name=f'TSR_Component_{i+1}')#self.poses[0], self.poses[1]
        return
    def Derivat(self,**kwargs):
        first_derivative = np.gradient(self.TT, axis=2)
        second_derivative = np.gradient(first_derivative, axis=2)
        self.mean_first_derivative = np.mean(first_derivative, axis=2)
        self.mean_second_derivative = np.mean(second_derivative, axis=2)
        return
    def SaveData_PCA_PPT(self,name='/PPT_PCA'):
        name=name+"_a="+str(self.a)+"_width="+str(self.width)
        save_dir = os.path.join(self.folder_path, "postprocessed_data",self.folder_name)
        os.makedirs(save_dir, exist_ok=True) 
        np.savez(save_dir+name+'.npz',
                  Phase=self.Phase[:,:,:26],Amp=self.Amp[:,:,:26],converted_data=self.converted_data)
        return
    def SaveData_TSR(self,name='/TSR'):
        name=name+"_a="+str(self.a)+"_width="+str(self.width)
        save_dir = os.path.join(self.folder_path, "postprocessed_data",self.folder_name)
        os.makedirs(save_dir, exist_ok=True) 
        np.savez(save_dir+name+'.npz',
                  TTSR=self.TTSR)
        return
    def SaveData_PCA(self,name='/PCA'):
        name=name+"_a="+str(self.a)+"_width="+str(self.width)
        save_dir = os.path.join(self.folder_path, "postprocessed_data",self.folder_name)
        os.makedirs(save_dir, exist_ok=True) 
        np.savez(save_dir+name+'.npz',
                  converted_data=self.converted_data)
        return
    def SaveData_ICA(self,name='/ICA'):
        name=name+"_a="+str(self.a)+"_width="+str(self.width)
        save_dir = os.path.join(self.folder_path, "postprocessed_data",self.folder_name)
        os.makedirs(save_dir, exist_ok=True) 
        np.savez(save_dir+name+'.npz',
                  ICA_data=self.ICA_data)
        return
    def SaveData_PPT(self,name='/PPT'):
        name=name+"_a="+str(self.a)+"_width="+str(self.width)
        save_dir = os.path.join(self.folder_path, "postprocessed_data",self.folder_name)
        os.makedirs(save_dir, exist_ok=True) 
        np.savez(save_dir+name+'.npz',
                  Phase=self.Phase[:,:,:26],Amp=self.Amp[:,:,:26],freqScale=self.freqScale)
        return
    def SaveData_Raw(self,name='/Raw'):
        name=name
        save_dir = os.path.join(self.folder_path, "postprocessed_data",self.folder_name)
        os.makedirs(save_dir, exist_ok=True) 
        np.savez(save_dir+name+'.npz',
                  Raw=self.TT)
        return
