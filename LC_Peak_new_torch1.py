import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from GRUD import GRUD
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
import seaborn as sns
from sys import platform
import logging
import os
local = True if platform != 'linux' else False
import gc

# get rid of the warnings
import warnings
warnings.filterwarnings("ignore")

# choose seed
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

scale_down = 7 # 7 days per time step

if local:
    file_name_for_cadence = "../cadence"
else:
    file_name_for_cadence = "../cadence_100000"
bandpasses = "ugrizy"

#For plot formatting
size = 13
plt.rc('font', size=size)          
plt.rc('axes', titlesize=size)     
plt.rc('axes', labelsize=size)   
plt.rc('xtick', labelsize=size)   
plt.rc('ytick', labelsize=size)    
plt.rc('legend', fontsize=size)    
plt.rc('figure', titlesize=size) 

tick_length_major = 7
tick_length_minor = 3
tick_width = 1

show = False

def get_observed_LC(cadence_index, cadence, time_bins, num_bands):
    """
    This function takes in a light curve at a fixed cadence and returns an observed light curve
    using LSST sampling and photometric noise.

    cadence_index: int, index of cadence file to use
    cadence: int, cadence of light curve in days
    time_bins: int, number of time bins in the light curve
    num_bands: int, number of bands in the light curve

    returns: numpy array, observed light curve, numpy array, photometric noise
    """
    JD_min = 60218
    time_list = []
    m_5s_list = []

    obs_mask = np.zeros((time_bins,num_bands))
    
    time_list = []

    for i,band in enumerate(bandpasses):
        file = f"{file_name_for_cadence}/sample_{cadence_index}_band_{band}_dates.dat"
        time = np.loadtxt(file, usecols=0)
        #m_5s = np.loadtxt(file, usecols=1)

        # add random shift in the time spacing. This is to avoid the season gaps all aligning.
        time -= JD_min

        time = (time/cadence).round().astype(int)
        time = np.unique(time)
        time_list.append(time)
    
    min_time = np.min([np.min(time) for time in time_list])
    max_time = np.max([np.max(time) for time in time_list])


    time_shift = np.random.randint(-min_time,max(time_bins-(max_time+1),1))

    for i in range(num_bands):

        time_val = time_list[i]+time_shift
        # make sure the time is within the bounds
        time_val = time_val[time_val >= 0] 
        time_val = time_val[time_val < time_bins]
        obs_mask[time_val,i] = 1.0
        
    return obs_mask


def apply_observation_sampling(light_curve, event, band_mask, num_bands, scale_down, rubin_sim, sample_cadence, sample_cadence_extra_band, cadence_index):
    """
    Apply observation sampling to the light curve. There are a few different options for observation sampling and multiple options can be used at once.

    light_curve: numpy array, light curve to be observed (time_bins, num_bands)
    event: numpy array, event label (time_bins)
    band_mask: numpy array, mask for the bands that are available (num_bands)
    num_bands: int, number of bands
    scale_down: int, number of days per time step
    rubin_sim: bool, whether to use the Rubin simulation
    sample_cadence: bool, whether to sample the cadence
    sample_cadence_extra_band: numpy array, mask for the extra band to sample the cadence
    cadence_index: int, index of the cadence file to use

    returns: numpy array, observed light curve, numpy array, observed event
    """

    event_mag_threshold = 1.0

    # get the regerence band. Use r band if it is available, otherwise use the last band available
    if band_mask[2] == 1:
        ref_band = 2
    else:
        ref_band = np.where(band_mask)[0][-1]

    if rubin_sim:

        # So we only include events with light curves that vary by at least 1 magnitude between its minimum and maximum
        if light_curve.max()-light_curve.min() < event_mag_threshold:
            event = np.zeros(event.shape)

        mask_rubin = get_observed_LC(cadence_index, scale_down, light_curve.shape[0], num_bands)

    min_season_length = 135
    max_season_length = 145
    n_visits = [57,72,186,194,169,174]
    
    # So we only include events with light curves that vary by at least 1 magnitude between its minimum and maximum
    if light_curve.max()-light_curve.min() < event_mag_threshold:
        event = np.zeros(event.shape)

    mask = np.ones(light_curve.shape)

    if sample_cadence:
        length_of_season = np.random.randint(min_season_length,max_season_length)
        offset = np.random.randint(365)

        for j in range(len(mask)):
            day = j*scale_down
            day_of_year = (day+offset) % 365 
            if 365-length_of_season < day_of_year < 365:
                mask[j,:] = 0
            else:
                #57 Observations in us
                #72 Observations in gs
                #186 Observations in rs
                #194 Observations in is
                #169 Observations in zs
                #174 Observations in ys
                
                for band in range(len(n_visits)):
                    p = 1-n_visits[band]/((365-(max_season_length+min_season_length)/2)*10/scale_down)
                    mask[j,band] = np.random.choice([0, 1],p=[p, 1-p])
        
    if rubin_sim:
        mask = np.expand_dims(np.array(sample_cadence_extra_band),axis=0) * mask + mask_rubin
        mask = np.clip(mask, 0.0 , 1.0)

    light_curve = mask*light_curve
    light_curve = light_curve - light_curve[:, ref_band][light_curve[:, ref_band] != 0.0][0]
    light_curve = mask*light_curve

    light_curve = light_curve * np.expand_dims(np.array(band_mask),axis=0)

    return light_curve, event

class Dataset_Loader(Dataset):
    def __init__(self, light_curve, event, cadence_file_name, scale_down, start_num_days, ends, days_after_to_check_event, days_after_factor, bias_num, augment, rubin_sim, sample_cadence, sample_cadence_extra_band, cadence_indexes, band_mask):
        """
        Pytorch dataset class for the light curve data loader.

        light_curve: numpy array, light curve to be observed (time_bins, num_bands)
        event: numpy array, event label (time_bins)
        cadence_file_name: str, name of the cadence file
        scale_down: int, number of days per time step
        start_num_days: int, number of days to start the light curve
        ends: int, number of days to end the light curve
        days_after_to_check_event: int, number of days after the event to check for a bias
        days_after_factor: float, factor that divides the days after to check event on the right side of the training label
        bias_num: int, number of biases to include in the training set
        augment: bool, whether to augment the data
        rubin_sim: bool, whether to use the Rubin simulation
        sample_cadence: bool, whether to sample the cadence
        sample_cadence_extra_band: numpy array, mask for the extra band to sample the cadence
        cadence_indexes: numpy array, indexes of the cadence files to use
        band_mask: numpy array, mask for the bands that are available (num_bands)
        """
        
        self.light_curve = light_curve
        self.event = event
        self.scale_down = scale_down
        self.start_num_days = start_num_days
        self.ends = ends
        self.days_after_to_check_event = days_after_to_check_event
        self.days_after_factor = days_after_factor
        self.bias_num = bias_num
        self.augment = augment
        self.rubin_sim = rubin_sim
        self.sample_cadence = sample_cadence
        self.sample_cadence_extra_band = sample_cadence_extra_band
        self.cadence_indexes = cadence_indexes
        self.band_mask = band_mask

        self.num_bands = light_curve.shape[-1]

    def __len__(self):
        return len(self.light_curve)
    
    def __getitem__(self, idx):

        # copy to avoid changing the original light curve, This is the light curve that will be observed

        light_curve = np.copy(self.light_curve[idx]) 
        light_curve_true = np.copy(light_curve)

        event = np.copy(self.event[idx])

        if self.augment:
            cadence_index = int(np.random.choice(self.cadence_indexes))
        else:
            cadence_index = int(self.cadence_indexes[idx % len(self.cadence_indexes)])

        light_curve, event = apply_observation_sampling(light_curve, event, self.band_mask, self.num_bands, scale_down, rubin_sim=self.rubin_sim, sample_cadence=self.sample_cadence, 
                                                        sample_cadence_extra_band=self.sample_cadence_extra_band, cadence_index=cadence_index)

        x = np.zeros(shape=(self.light_curve.shape[1],self.light_curve.shape[2]+1))

        k = 0
        while k < self.bias_num:
            start = np.random.randint(self.start_num_days,self.light_curve.shape[1]-self.ends)
            
            if 1.0 in self.event[idx,start-self.days_after_to_check_event:start+int(self.days_after_to_check_event//self.days_after_factor)]:
                k = self.bias_num
            else:
                k+=1

            x[:start,:self.num_bands] = light_curve[:start] 
            x[:start,self.num_bands] = 1.0

        if 1.0 in event[start-self.days_after_to_check_event:start+int(self.days_after_to_check_event//self.days_after_factor)]:
            y = 1.0
        else:
            y = 0.0

        # Change arrays to np.float32 to convert to torch tensors
        x = x.astype(np.float32)
        y = np.array([y]).astype(np.float32)
        light_curve_true = light_curve_true.astype(np.float32)
        light_curve = light_curve.astype(np.float32)
        event = event.astype(np.float32)

        # make sure the light curve is not nan
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        light_curve_true = np.nan_to_num(light_curve_true, nan=0.0, posinf=0.0, neginf=0.0)
        light_curve = np.nan_to_num(light_curve, nan=0.0, posinf=0.0, neginf=0.0)
        event = np.nan_to_num(event, nan=0.0, posinf=0.0, neginf=0.0)

        sample = {'x': x, 'y': y, 'light_curve_true': light_curve_true, 'light_curve_obs': light_curve, 'event': event}

        return sample

def get_edges(lst):
    """
    Helper function to get the edges of a sequence of 1s in an array.

    lst: numpy array, array of 1s and 0s

    returns: numpy array, array of edges
    """
    diff = np.diff(lst)
    edges = np.where(diff != 0)[0] # Get the indices where the difference is non-zero
    if lst[0] == 1:  # if the sequence starts with 1, add 0 as the starting edge
        edges = np.concatenate(([0], edges))
    
    if len(edges) % 2 != 0:  # if the sequence ends with 1, add last index as the ending edge
        edges = np.concatenate((edges, [len(lst) - 1]))
    
    return edges.reshape(-1, 2)  # reshape the array into pairs

@torch.no_grad()
def prediction_curve(model, num, curve, event, scale_down, start_num_days, ends, days_after_to_check_event, days_after_factor, device, save_plot=True, save_path='results'):
    """
    Function used to make predictions on a light curve and plot the results.

    model: pytorch model, trained model
    num: int, number of the light curve
    curve: numpy array, light curve to make predictions on (time_bins, num_bands)
    event: numpy array, event label (time_bins)
    scale_down: int, number of days per time step
    start_num_days: int, number of days to start the light curve
    ends: int, number of days to end the light curve
    days_after_to_check_event: int, number of days after the event to check for a bias
    days_after_factor: float, factor that divides the days after to check event on the right side of the training label
    device: torch.device, device to use for the model
    save_plot: bool, whether to save the plot
    save_path: str, path to save the plot
    """
    time_bins = curve.shape[0]
    num_bands = curve.shape[1]

    threshold = 0.5

    prediction_list = []

    '''
    for i in range(start_num_days,time_bins-days_after_to_check_event): #range(start_num_days,light_curve_val.shape[1]-days_after_to_check_event)
        
        LC_section = np.zeros((time_bins, num_bands+1))
        LC_section[:i, :num_bands] = curve[:i] 
        LC_section[:i, num_bands] = 1.0
        LC_section = np.expand_dims(LC_section, axis=0).astype(np.float32)
        
        # change tensor to float
        prediction = model(torch.from_numpy(LC_section).to(device)).squeeze().detach().cpu().item()
        prediction_list.append(prediction)

    prediction = np.array(prediction_list)
    del prediction_list
    '''
    LC_batch = []
    for i in range(start_num_days,time_bins-days_after_to_check_event): 
        LC_section = np.zeros((time_bins, num_bands+1))
        LC_section[:i, :num_bands] = curve[:i] 
        LC_section[:i, num_bands] = 1.0
        LC_section = LC_section.astype(np.float32)
        LC_batch.append(LC_section)

    # combine into shape (batch_size, time_bins, num_bands+1) where now the batch is size (time_bins-days_after_to_check_event-start_num_days)
    LC_batch = np.stack(LC_batch, axis=0)
    prediction = model(torch.from_numpy(LC_batch).to(device)).squeeze().detach().cpu().numpy()

    prediction_flag = (prediction >= threshold).astype(float)
    
    #alert system
    green_trigger = 50
    yellow_trigger = 100
    red_trigger = 150
    
    length_above_threshold = np.zeros(len(prediction_flag))
    for i in range(len(prediction_flag)):
        if prediction_flag[i] == 1.0:
            if i > 0:
                length_above_threshold[i] = length_above_threshold[i-1]+scale_down
        else:
            length_above_threshold[i] = 0
    
    ##plot labels##
    # Find the indices of the 1s in the array
    ones_indices = np.where(event == 1.0)[0]

    # Find the indices of the transitions between zeros and ones
    transitions = np.where(np.diff(ones_indices) != 1)[0] + 1

    # Split the array into subarrays corresponding to each plateau of 1s
    plateaus = np.split(ones_indices, transitions)

    # Compute the mean position of each plateau
    mean_positions = [np.mean(p) for p in plateaus]
    
    # only include the event predicted in the "red alert" region
    green_prediction_flag = (length_above_threshold >= scale_down).astype(float)
    red_prediction_flag = (length_above_threshold > yellow_trigger).astype(float)

    correct_pred = 0 
    incorrect_pred = 0
    
    correct_pred_red = 0
    correct_pred_red_in_label = 0

    for event_pos in mean_positions:
        if start_num_days+days_after_to_check_event < event_pos < (time_bins-days_after_to_check_event):
            if green_prediction_flag[int(event_pos)-start_num_days] == 1.0:
                correct_pred += 1
            else:
                incorrect_pred += 1

            if red_prediction_flag[int(event_pos)-start_num_days] == 1.0:
                correct_pred_red += 1

            if 1.0 in red_prediction_flag[int(event_pos)-start_num_days-days_after_to_check_event:int(event_pos)-start_num_days+int(days_after_to_check_event//days_after_factor)]:
                correct_pred_red_in_label += 1


    false_positive = 0
    edges_red = get_edges(red_prediction_flag)
    edges_green = get_edges(green_prediction_flag)

    for i in range(edges_green.shape[0]):
        false_positive_flag = True

        # Check if there is an event in the green zone, only a false positive if there is no event in the green zone
        for event_pos in mean_positions:
            if (edges_green[i,0] <= event_pos-start_num_days <= edges_green[i,1]): 
                false_positive_flag = False
        
        # Check if there is a red zone in the green zone, otherwise we do not count it as a false positive
        red_in_green_flag = False
        for j in range(edges_red.shape[0]):
            # Check if there is a red zone in the green zone
            if (edges_green[i,0] <= edges_red[j,0] <= edges_green[i,1]) or (edges_green[i,0] <= edges_red[j,1] <= edges_green[i,1]):
                red_in_green_flag = True

        # Only count false positive if the right edge of the green zone is not to close to the beginning of the light curve and the left edge is not too close to the end
        not_near_edge = True
        if (edges_green[i,0] < days_after_to_check_event) or (edges_green[i,1] > time_bins-2*days_after_to_check_event):
            near_edge_flag = False

        # Only if there is no event in the green zone and we predict red do we count as a false positive
        if false_positive_flag and red_in_green_flag and not_near_edge:
            false_positive += 1

    if save_plot:
        fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(12,7),gridspec_kw={'height_ratios': [1,1.5,3]},sharex=True)

        ax1.set_ylabel('alert')
        #black out the part we do not make predictions for
        ax1.axvspan(0, start_num_days*scale_down, color='black', alpha=0.1)
        
        t = start_num_days*scale_down+np.linspace(0,scale_down*len(prediction_flag)-1,len(prediction_flag))
        
        #This will set the axis but will not show up on the plot
        ax1.plot(t,np.zeros(len(t))-1)
        
        ax1.fill_between(t,1,0,where=(length_above_threshold>scale_down) & (length_above_threshold<=green_trigger),
                        color='green',alpha=0.8)
        ax1.fill_between(t,1,0,where=(length_above_threshold>green_trigger) & (length_above_threshold<=yellow_trigger),
                        color='yellow',alpha=0.8)
        ax1.fill_between(t,1,0,where=(length_above_threshold>yellow_trigger),
                        color='red',alpha=0.8)
        ax1.set_ylim(0,1)
        ax1.set_yticks([])
        ax1.set_xlim(0,scale_down*time_bins-1)
        ax1.minorticks_on()
        ax1.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        ax1.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)

        labels = ['u', 'g', 'r' , 'i', 'z', 'y','crossing']

        colors = ['violet', 'g', 'r' , 'brown', 'grey', 'black']

        
        ax2.set_ylim(0,1.05)
        ax2.set_ylabel('probability')
        
        
        ax2.axvspan(0, start_num_days*scale_down, color='black', alpha=0.1)
        ax2.axvspan((time_bins-days_after_to_check_event)*scale_down,scale_down*time_bins-1, 
                    color='black', alpha=0.1)

        ax2.plot(t, prediction, label='pred')
        

        ax2.plot(np.linspace(0,scale_down*time_bins-1,time_bins), event, label='peak')
        

        label = np.zeros(event.shape)
        try:
            for mean in mean_positions:
                min_val = int(np.clip(mean-days_after_to_check_event,0,time_bins))
                max_val = int(np.clip(mean+days_after_to_check_event//days_after_factor,0,time_bins))

                label[min_val:max_val]=1.0
            ax2.plot(np.linspace(0,scale_down*time_bins-1,time_bins), label, label='label')
        except:
            pass
        ax2.tick_params(axis='y')
        ax2.set_xlim(0,scale_down*time_bins-1)
        ax2.minorticks_on()
        ax2.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        ax2.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        ax2.legend(fontsize=10.5,loc="upper left")

        
        for i in range(num_bands):
            light_curve_temp = np.copy(curve[:,i])
            light_curve_temp[light_curve_temp==0]=np.nan

            ax3.plot(np.linspace(0,scale_down*time_bins-1,time_bins), light_curve_temp, 'o', color=colors[i],label=labels[i],markersize=2)

        ax3.set_xlabel('time [days]')
        ax3.set_ylabel('relative magnitude')
        ax3.set_xlim(0,scale_down*time_bins-1)
        ax3.minorticks_on()
        ax3.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        ax3.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        ax3.tick_params(axis='y')
        ax3.legend(fontsize=13)

        
        fig.tight_layout()
        
        plt.savefig(f'{save_path}/recovery/example_output_{num}.pdf',bbox_inches='tight')
        plt.close()

    return correct_pred, incorrect_pred, correct_pred_red, correct_pred_red_in_label, false_positive, prediction


# Create the model in pytorch
class NN(nn.Module):
    def __init__(self, hidden_size, input_size, device, mask=True, band_mask=None):
        super(NN, self).__init__()
        """
        Class for the neural network model. The model is bi-directional GRU-D/GRU with 3 layers and a 3 layer fully connected network at the end.

        hidden_size: int, size of the hidden layer
        input_size: int, size of the input layer
        device: torch.device, device to use for the model
        mask: bool, whether to use the GRU-D model
        band_mask: list, mask for the bands that are available
        """
        if band_mask is not None:
            self.use_band_mask = True
            input_size = int(sum(band_mask))+1
            # only use the bands that are not masked out. The last column is the mask so we always include it
            self.band_mask = torch.tensor(band_mask+[True]).type(torch.bool).to(device)
        else:
            self.use_band_mask = False
            self.band_mask = None

        if mask:
            self.grud = GRUD(input_size=input_size, hidden_size=hidden_size, device=device)
        else:
            self.grud = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
            
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.gru1 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

        if mask:
            self.grud_flipped = GRUD(input_size=input_size, hidden_size=hidden_size, device=device)
        else:
            self.grud_flipped = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.layer_norm1_flipped = nn.LayerNorm(hidden_size)
        self.gru1_flipped = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.layer_norm2_flipped = nn.LayerNorm(hidden_size)
        self.gru2_flipped = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.layer_norm3_flipped = nn.LayerNorm(hidden_size)

        self.fc1 = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(2*hidden_size),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(2*hidden_size),
        )

        self.fc3 = nn.Linear(2*hidden_size, 1)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        if self.use_band_mask:
            # Only use the bands that are not masked out
            x = x[:, :, self.band_mask]

        x_flipped = torch.flip(x, dims=(1,)) # Flip the input along the time axis

        x, _ = self.grud(x)
        x = self.layer_norm1(x)

        skip = x

        x, _ = self.gru1(x)
        x = self.layer_norm2(x)

        x = x + skip
        skip = x

        x, _ = self.gru2(x)
        x = self.layer_norm3(x)
        
        x = x + skip

        x_flipped, _ = self.grud_flipped(x_flipped)
        x_flipped = self.layer_norm1_flipped(x_flipped)

        skip = x_flipped

        x_flipped, _ = self.gru1_flipped(x_flipped)
        x_flipped = self.layer_norm2_flipped(x_flipped)

        x_flipped = x_flipped + skip
        skip = x_flipped

        x_flipped, _ = self.gru2_flipped(x_flipped)
        x_flipped = self.layer_norm3_flipped(x_flipped)

        x_flipped = x_flipped + skip
        del skip

        # Only take the last time step and concatenate the forward and backward pass
        x = torch.cat((x[:,-1,:], x_flipped[:,-1,:]), dim=1)
        del x_flipped

        skip = x
        x = self.fc1(x)
        x = x + skip
        x = self.fc2(x)

        # Sigmoid activation function, output is between 0 and 1 for binary classification
        x = self.sigmoid(self.fc3(x))
    
        return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

def main(
    learning_rate_initial = 2.e-3, # 2e-3
    batch_size = 512, # 512
    hidden_size = 128, # 128
    epochs = 0,  # 250
    fraction = 1.0,
    validation_rate = 0.1,
    rubin_sim=True, 
    sample_cadence=True,
    sample_cadence_extra_band=[0,0,0,0,0,0],
    band_mask=[1,1,1,1,1,1],
    final_test_size=1000, 
    grad_clip_value=1.0,
    save_path = 'results1',
    load_model = False,
    load_path = "results1/my_model.pth",
):
    assert type(learning_rate_initial) == float and learning_rate_initial > 0, "learning_rate_initial must be a positive float"
    assert type(batch_size) == int and batch_size > 0, "batch_size must be a positive integer"
    assert type(hidden_size) == int and hidden_size > 0, "hidden_size must be a positive integer"
    assert type(epochs) == int and epochs >= 0, "epochs must be a non-negative integer"
    assert type(fraction) == float and 0 < fraction <= 1, "fraction must be a float between 0 and 1"
    assert type(validation_rate) == float and 0 < validation_rate < 1, "validation_rate must be a float between 0 and 1"
    assert type(rubin_sim) == bool, "rubin_sim must be a boolean"
    assert type(sample_cadence) == bool, "sample_cadence must be a boolean"
    assert type(sample_cadence_extra_band) == list and len(sample_cadence_extra_band) == 6, "sample_cadence_extra_band must be a list of length 6"
    assert all(type(i) == int and i in [0, 1] for i in sample_cadence_extra_band), "sample_cadence_extra_band must be a list of 0s and 1s"
    assert type(band_mask) == list and len(band_mask) == 6, "band_mask must be a list of length 6"
    assert all(type(i) == int and i in [0, 1] for i in band_mask), "band_mask must be a list of 0s and 1s"
    assert type(final_test_size) == int and final_test_size > 0, "final_test_size must be a positive integer"
    assert type(grad_clip_value) == float and grad_clip_value > 0, "grad_clip_value must be a positive float"
    assert type(save_path) == str, "save_path must be a string"
    assert type(load_model) == bool, "load_model must be a boolean"
    assert type(load_path) == str, "load_path must be a string"

    if rubin_sim:
        mask_obs = True
    elif sample_cadence:
        mask_obs = True
    else:
        mask_obs = False
        
    # see if gpu is working
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # see type of gpu
    if device.type == 'cuda':
        print(torch.cuda.get_device_name())

    num_cpu = os.cpu_count()
    print(f'Number of CPUs: {num_cpu}')

    #make directory to save results
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f'{save_path}/recovery', exist_ok=True)
    os.makedirs(f'{save_path}/recovery/saved_output', exist_ok=True)

    name = "input_weekly.fits"

    num_bands = 6 
    label_start = num_bands

    with fits.open(name,memmap=False) as hdul:
        # Light curve
        light_curve = hdul[0].data[:,:,:num_bands]
        # Event labels
        event = hdul[0].data[:,:,label_start]

    gc.collect()

    print("light curve shape")
    print(light_curve.shape)
    print("event shape")
    print(event.shape)

    num_bands = light_curve.shape[-1]

    randomize = np.arange(len(light_curve))
    np.random.seed(0) # set seed for reproducibility of the test set
    np.random.shuffle(randomize)
    
    light_curve = light_curve[randomize][:int(fraction*len(light_curve))]
    event = event[randomize][:int(fraction*len(event))]
    randomize = randomize[:int(fraction*len(randomize))]
    gc.collect()

    print("Using light curve of shape")
    print(light_curve.shape)
    print("Using event of shape")
    print(event.shape)

    event[event>0.0] = 1.0
    event = np.clip(event,0.0,1.0)

    start_num_days = 600
    days_after_to_check_event = 150
    days_after_factor = 1.0
    ends = days_after_to_check_event*days_after_factor

    # Factor that divides the days after to check event on the right side of the training label.
    
    shape = light_curve.shape

    #converts to bins instead of days
    ends = ends // scale_down
    start_num_days = start_num_days // scale_down
    days_after_to_check_event = days_after_to_check_event // scale_down

    num_val = int(validation_rate*len(light_curve))
    num_test = int(validation_rate*len(light_curve))

    light_curve_val = light_curve[:num_val]
    event_val = event[:num_val]

    light_curve_test = light_curve[num_val:num_val+num_test]
    event_test = event[num_val:num_val+num_test]
    test_indexes = randomize[num_val:num_val+num_test]

    light_curve_train = light_curve[num_val+num_test:]
    event_train = event[num_val+num_test:]

    try:
        del light_curve
        del event
    except:
        pass

    print("light curve train shape")
    print(light_curve_train.shape)

    print("light curve val shape")
    print(light_curve_val.shape)

    print("light curve test shape")
    print(light_curve_test.shape)

    if load_model:
        print(f'Loading model from {load_path}')
        model = NN(hidden_size=hidden_size, input_size=num_bands+1, device=device, mask=mask_obs, band_mask=band_mask).to(device)
        model_old = torch.load(load_path).to(device)
        model.load_state_dict(model_old.state_dict(), strict=False)
        del model_old
    else:
        # input_size is the number of bands plus 1 for the mask
        model = NN(hidden_size=hidden_size, input_size=num_bands+1, device=device, mask=mask_obs, band_mask=band_mask).to(device)

    # print the number of parameters in the model
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_initial)
    # exponential decay of learning rate by a factor of 0.1 over the course of the training
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(0.1)/epochs))

    cadence_files = glob.glob(f'{file_name_for_cadence}/*.dat')
    num_cadence_files = int(len(cadence_files)//num_bands)

    training_cadence_indexes = np.arange(num_cadence_files*(1.0-2.0*validation_rate))
    validation_cadence_indexes = np.arange(num_cadence_files*(1.0-2.0*validation_rate),num_cadence_files*(1.0-validation_rate))
    test_cadence_indexes = np.arange(num_cadence_files*(1.0-validation_rate),num_cadence_files)

    #ight_curve, event, cadence_file_name, scale_down, start_num_days, ends, days_after_to_check_event, days_after_factor, bias_num
    training_dataset = Dataset_Loader(light_curve=light_curve_train, event=event_train, cadence_file_name=None, scale_down=scale_down, 
                                      start_num_days=start_num_days, ends=ends, days_after_to_check_event=days_after_to_check_event, 
                                      days_after_factor=days_after_factor, bias_num=1, augment=True, rubin_sim=rubin_sim, 
                                      sample_cadence=sample_cadence, sample_cadence_extra_band=sample_cadence_extra_band,
                                      cadence_indexes=training_cadence_indexes, band_mask=band_mask)
    
    validation_dataset = Dataset_Loader(light_curve=light_curve_val, event=event_val, cadence_file_name=None, scale_down=scale_down,
                                        start_num_days=start_num_days, ends=ends, days_after_to_check_event=days_after_to_check_event,
                                        days_after_factor=days_after_factor, bias_num=1, augment=False, rubin_sim=rubin_sim, 
                                        sample_cadence=sample_cadence, sample_cadence_extra_band=sample_cadence_extra_band,
                                        cadence_indexes=validation_cadence_indexes, band_mask=band_mask)
    
    test_dataset = Dataset_Loader(light_curve=light_curve_test, event=event_test, cadence_file_name=None, scale_down=scale_down,
                                    start_num_days=start_num_days, ends=ends, days_after_to_check_event=days_after_to_check_event,
                                    days_after_factor=days_after_factor, bias_num=1, augment=False, rubin_sim=rubin_sim, 
                                    sample_cadence=sample_cadence, sample_cadence_extra_band=sample_cadence_extra_band,
                                    cadence_indexes=test_cadence_indexes, band_mask=band_mask)

    
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=max(num_cpu-1,1), pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=max(num_cpu-1,1), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=max(num_cpu-1,1), pin_memory=True)

    # Define the loss function
    criterion = nn.BCELoss()

    # Training loop
    loss_epoch = []
    accuracy_epoch = []
    precision_epoch = []
    recall_epoch = []
    f1_epoch = []

    loss_val_epoch = []
    accuracy_val_epoch = []
    precision_val_epoch = []
    recall_val_epoch = []
    f1_val_epoch = []

    for epoch in tqdm(range(epochs)):
        model.train()

        loss_train = 0.0
        accuracy_train = 0.0
        precision_train = 0.0
        recall_train = 0.0
        f1_train = 0.0

        for i, data in enumerate(train_loader):
            x = data['x'].to(device)
            y = data['y'].to(device)

            model.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()

            # Add gradient clipping to prevent exploding gradients
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)

            # Check if the gradients are finite and not NaN
            nan_in_grad = False
            for name, param in model.named_parameters():
                if torch.isnan(param.grad).any():
                    print(f'Gradient for {name} is nan')
                    nan_in_grad = True
                    
            if not nan_in_grad:
                optimizer.step()

            del loss
            y_pred = outputs.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

            if i == 0:
                y_pred_train = y_pred
                y_true_train = y_true
            else:
                y_pred_train = np.concatenate((y_pred_train, y_pred), axis=0)
                y_true_train = np.concatenate((y_true_train, y_true), axis=0)

        # get the number of events in the training set as a fraction of the total number of events
        print(f'Fraction of events in training set: {100.0*np.sum(y_true_train)/len(y_true_train):.2f}%')


        loss_train = log_loss(y_true_train, y_pred_train)
        # Round the predictions to 0 or 1 for classification
        y_pred_train = y_pred_train.round() 
        accuracy_train = accuracy_score(y_true_train, y_pred_train)
        precision_train = precision_score(y_true_train, y_pred_train)
        recall_train = recall_score(y_true_train, y_pred_train)
        f1_train = f1_score(y_true_train, y_pred_train)
        del y_pred_train, y_true_train

        gc.collect()
        print(f'Epoch {epoch+1}, Training loss: {loss_train:.4f}, Training accuracy: {accuracy_train:.4f}, Training precision: {precision_train:.4f}, Training recall: {recall_train:.4f}, Training F1: {f1_train:.4f}')
        
        loss_epoch.append(loss_train)
        accuracy_epoch.append(accuracy_train)
        precision_epoch.append(precision_train)
        recall_epoch.append(recall_train)
        f1_epoch.append(f1_train)

        scheduler.step()

        with torch.no_grad():
            model.zero_grad()
            model.eval()

            loss_val = 0.0
            accuracy_val = 0.0
            precision_val = 0.0
            recall_val = 0.0
            f1_val = 0.0

            for i, data in enumerate(val_loader):
                x = data['x'].to(device)
                y = data['y'].to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                y_pred = outputs.cpu().detach().numpy()
                y_true = y.cpu().detach().numpy()

                if i == 0:
                    y_pred_val = y_pred
                    y_true_val = y_true
                else:
                    y_pred_val = np.concatenate((y_pred_val, y_pred), axis=0)
                    y_true_val = np.concatenate((y_true_val, y_true), axis=0)
            
            loss_val = log_loss(y_true_val, y_pred_val)
            y_pred_val = y_pred_val.round()
            accuracy_val = accuracy_score(y_true_val, y_pred_val)
            precision_val = precision_score(y_true_val, y_pred_val)
            recall_val = recall_score(y_true_val, y_pred_val)
            f1_val = f1_score(y_true_val, y_pred_val)

            del y_pred_val, y_true_val
            gc.collect()

            loss_val_epoch.append(loss_val)
            accuracy_val_epoch.append(accuracy_val)
            precision_val_epoch.append(precision_val)
            recall_val_epoch.append(recall_val)
            f1_val_epoch.append(f1_val)

            print(f'Epoch {epoch+1}, Validation loss: {loss_val:.4f}, Validation accuracy: {accuracy_val:.4f}, Validation precision: {precision_val:.4f}, Validation recall: {recall_val:.4f}, Validation F1: {f1_val:.4f}')

    # Save model
    torch.save(model, f'{save_path}/my_model.pth')

    # Test the model
    with torch.no_grad():
        model.eval()

        loss_test = 0.0
        accuracy_test = 0.0
        precision_test = 0.0
        recall_test = 0.0
        f1_test = 0.0

        for i, data in enumerate(test_loader):
            x = data['x'].to(device)
            y = data['y'].to(device)
            outputs = model(x)
            loss = criterion(outputs, y) 
            y_pred = outputs.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

            if i == 0:
                y_pred_test = y_pred
                y_true_test = y_true
            else:
                y_pred_test = np.concatenate((y_pred_test, y_pred), axis=0)
                y_true_test = np.concatenate((y_true_test, y_true), axis=0)

        loss_test = log_loss(y_true_test, y_pred_test)
        y_pred_test = y_pred_test.round()
        accuracy_test = accuracy_score(y_true_test, y_pred_test)
        precision_test = precision_score(y_true_test, y_pred_test)
        recall_test = recall_score(y_true_test, y_pred_test)
        f1_test = f1_score(y_true_test, y_pred_test)

        print(f'Test loss: {loss_test:.4f}, Test accuracy: {accuracy_test:.4f}, Test precision: {precision_test:.4f}, Test recall: {recall_test:.4f}, Test F1: {f1_test:.4f}')

        # Compute the confusion matrix
        cm = confusion_matrix(y_true_test, y_pred_test)

        # Plot the confusion matrix
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', cbar=False,
                        xticklabels=['No Peak', 'Peak'], yticklabels=['No Peak', 'Peak'],
                        annot_kws={'size': 14})
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Truth', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        plt.savefig(f"{save_path}/confusion_matrix.pdf", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        # Normalize the confusion matrix
        cm = confusion_matrix(y_true_test, y_pred_test, normalize='all')

        # Plot the normalized confusion matrix
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', cbar=False,
                        xticklabels=['No Peak', 'Peak'], yticklabels=['No Peak', 'Peak'],
                        annot_kws={'size': 14})
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Truth', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        plt.savefig(f"{save_path}/confusion_matrix_norm.pdf", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        # Plot the loss and accuracy vs epoch
        if epochs > 0:
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, epochs+1), loss_epoch, label='training')
            plt.plot(range(1, epochs+1), loss_val_epoch, label='validation')
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('BCE', fontsize=14)
            plt.legend()
            plt.savefig(f"{save_path}/loss_vs_epoch.pdf", bbox_inches='tight')
            plt.close()
        
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, epochs+1), accuracy_epoch, label='training')
            plt.plot(range(1, epochs+1), accuracy_val_epoch, label='validation')
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('accuracy', fontsize=14)
            plt.legend()
            plt.savefig(f"{save_path}/accuracy_vs_epoch.pdf", bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.plot(range(1, epochs+1), precision_epoch, label='training')
            plt.plot(range(1, epochs+1), precision_val_epoch, label='validation')
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('precision', fontsize=14)
            plt.legend()
            plt.savefig(f"{save_path}/precision_vs_epoch.pdf", bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.plot(range(1, epochs+1), recall_epoch, label='training')
            plt.plot(range(1, epochs+1), recall_val_epoch, label='validation')
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('recall', fontsize=14)
            plt.legend()
            plt.savefig(f"{save_path}/recall_vs_epoch.pdf", bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.plot(range(1, epochs+1), f1_epoch, label='training')
            plt.plot(range(1, epochs+1), f1_val_epoch, label='validation')
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('F1 score', fontsize=14)
            plt.legend()
            plt.savefig(f"{save_path}/f1_vs_epoch.pdf", bbox_inches='tight')
            plt.close()

        final_test_size = min(final_test_size,event_test.shape[0])
        num_save = 50
        saved = 0
        correct_pred_tot, incorrect_pred_tot, correct_pred_red_tot, correct_pred_red_in_label_tot, false_positive_tot = 0, 0, 0, 0, 0

        save_output_prediciton = True

        count = 0
        # use tqdm to show a progress bar with count
        progress = tqdm(total=final_test_size, desc="Analyzing test set")

        for batch, data in enumerate(test_loader):

            light_curve = data['light_curve_obs'].cpu().detach().numpy()
            light_curve_true = data['light_curve_true'].cpu().detach().numpy()
            event = data['event'].cpu().detach().numpy()

            # loop over the batch one light curve at a time
            for i in range(light_curve.shape[0]):

                light_curve_val = light_curve[i]
                light_curve_true_val = light_curve_true[i]
                event_val = event[i]

                # check if there is a peak in the light curve so we save the plot. 
                # Always save the first 10 plots to include some that have no peaks as well.
                if (1.0 in event_val) or (count < 10):
                    save_plot = True if saved < num_save else False
                    saved += 1
                else:
                    save_plot = False 

                correct_pred, incorrect_pred, correct_pred_red, correct_pred_red_in_label, false_positive, prediction = prediction_curve(model, count, light_curve_val, event_val,
                                                                                                                            scale_down, start_num_days, ends, days_after_to_check_event, 
                                                                                                                            days_after_factor, device, save_plot=save_plot, save_path=save_path)
                correct_pred_tot += correct_pred
                incorrect_pred_tot += incorrect_pred
                correct_pred_red_tot += correct_pred_red
                correct_pred_red_in_label_tot += correct_pred_red_in_label
                false_positive_tot += false_positive

                if save_output_prediciton:
                    # Save the output prediction as a fits file
                    primary_hdu = fits.PrimaryHDU()
                    primary_hdu.header['Index'] = test_indexes[count]
                    
                    hdu1 = fits.ImageHDU(light_curve_val, name='light_curve_obs')
                    hdu2 = fits.ImageHDU(light_curve_true_val, name='light_curve_true')
                    hdu3 = fits.ImageHDU(event_val, name='event')
                    hdu4 = fits.ImageHDU(prediction, name='prediction')


                    # Combine the HDUs into a single HDU list
                    hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4])

                    hdul.writeto(f'{save_path}/recovery/saved_output/output_{count}.fits', overwrite=True)

                count += 1
                progress.update(1)

                print(f"Test samples analyzed {count}")

                if count >= final_test_size:
                    break    
            
            if count >= final_test_size:
                break

        del light_curve, light_curve_true, event

        # close the progress bar
        progress.close()

        print(f"Test samples analyzed {final_test_size}")

        print(f"Total correct predictions: {correct_pred_tot}")
        print(f"Total incorrect predictions: {incorrect_pred_tot}")
        print(f"Total correct predictions in red zone: {correct_pred_red_tot}")
        print(f"Total predictions of red zone in label: {correct_pred_red_in_label_tot}")
        print(f"Total false positives: {false_positive_tot}")

        print(f"Frac correct predictions out of total events: {round(100*correct_pred_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%")
        print(f"Frac incorrect predictions out of total events: {round(100*incorrect_pred_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%")
        print(f"Frac correct predictions in red zone out of total events: {round(100*correct_pred_red_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%")
        print(f"Frac predictions of red zone in label out of total events: {round(100*correct_pred_red_in_label_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%")
        print(f"Frac false positives out of total events: {round(100*false_positive_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%")
        print(f"Frac false positives out of num val light curves : {round(100*false_positive_tot/num_test, 3)}%")


        # Save metrics to a file
        with open(f"{save_path}/metrics_test.txt", "w") as file:
            
            if epochs > 0:
                # Training metrics
                file.write(f'Training loss: {loss_train:.4f}, Training accuracy: {accuracy_train:.4f}, Training precision: {precision_train:.4f}, Training recall: {recall_train:.4f}, Training F1: {f1_train:.4f}\n')
                # Validation metrics
                file.write(f'Validation loss: {loss_val:.4f}, Validation accuracy: {accuracy_val:.4f}, Validation precision: {precision_val:.4f}, Validation recall: {recall_val:.4f}, Validation F1: {f1_val:.4f}\n')
            # Test metrics
            file.write(f'Test loss: {loss_test:.4f}, Test accuracy: {accuracy_test:.4f}, Test precision: {precision_test:.4f}, Test recall: {recall_test:.4f}, Test F1: {f1_test:.4f}\n')


            file.write(f"Test samples analyzed {num_test}\n")
            file.write(f"Total correct predictions: {correct_pred_tot}\n")
            file.write(f"Total incorrect predictions: {incorrect_pred_tot}\n")
            file.write(f"Total correct predictions in red zone: {correct_pred_red_tot}\n")
            file.write(f"Total predictions of red zone in label: {correct_pred_red_in_label_tot}\n")
            file.write(f"Total false positives: {false_positive_tot}\n")

            file.write(f"Frac correct predictions out of total events: {round(100*correct_pred_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%\n")
            file.write(f"Frac incorrect predictions out of total events: {round(100*incorrect_pred_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%\n")
            file.write(f"Frac correct predictions in red zone out of total events: {round(100*correct_pred_red_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%\n")
            file.write(f"Frac predictions of red zone in label out of total events: {round(100*correct_pred_red_in_label_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%\n")
            file.write(f"Frac false positives out of total events: {round(100*false_positive_tot/(correct_pred_tot+incorrect_pred_tot), 3)}%\n")
            file.write(f"Frac false positives out of num val light curves : {round(100*false_positive_tot/num_test, 3)}%\n")

if __name__ == "__main__":
    main()
    print("Done!")
