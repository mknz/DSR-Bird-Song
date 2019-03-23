import os
from urllib.request import urlcleanup
from .utils.balanced_split import make_split
from .utils import sql_selectors
from .precomputing_slices import Slicer
from .selections import Selection
from collections import Counter
import pandas as pd
import sqlite3
import datetime
import matplotlib.pyplot as plt

class DatabaseManager(object):
    """ This class bundles the various functions for acquiring, inventorizing, 
    manipulating and serving data and exposes them as easily usable methods."""
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        self.Selection = Selection
        self.signal_dir = os.path.join(storage_dir, 'signal_slices')
        self.noise_dir = os.path.join(storage_dir, 'noise_slices')
        
        if not os.path.isdir(storage_dir):
            print('Creating empty directory.')
            os.mkdir(storage_dir)
        if not os.path.isdir(self.signal_dir):
            os.mkdir(self.signal_dir)
        if not os.path.isdir(self.noise_dir):
            os.mkdir(self.noise_dir)
        if not os.path.isfile(os.path.join(storage_dir, 'db.sqlite')):
            print('No SQL database built yet, initiating one.')
            #TODO: Actually do that and include all the steps from 
            # database creation so that this entire package could be ported 
            # somewhere else and rebuilt with one command
        
        self.conn = sqlite3.connect(os.path.join(storage_dir, 'db.sqlite'))    

        self.SignalSlicer = Slicer(self.signal_dir, type='signal')
        self.NoiseSlicer = Slicer(self.noise_dir, type='noise')
    
    def make_selection(self, nr_of_classes=100, slices_per_class=1200):
        self.Selection = Selection(self.conn, nr_of_classes, slices_per_class)

        
    def selection_df(self):
        """ Based on the classes in the current selection return a dataframe with 
        label and path for each slice available for these classes. Alert the user 
        if additional slices could be downloaded to complete the selection. """
        classes_in_selection = self.Selection.classes_in_selection
        all = self.inventory_df()
        available_in_selection = all[all.label.isin(classes_in_selection)].reset_index(drop=True)
        ideal = self.Selection.nr_of_classes * self.Selection.slices_per_class
        slices_available = available_in_selection.groupby('label').count().sum().values
        if slices_available < ideal:
            print(f"We are {ideal - slices_available} slices short of the Selection. \
            You can call the method 'download_missing' to fill them up if more are available.")    
        return available_in_selection
    
    def inventory_df(self):
        """ Retrieves class name for each slice currently in signal_dir 
        and returns of df with the file name for each recording and its 
        associated label. """
        list_recs = []
        for file in os.listdir(self.signal_dir):
            if file.endswith('.pkl'):
                rec_id = file.split('_')[0]
                species = sql_selectors.species_by_rec_id(self.conn, rec_id)
                list_recs.append((file, species))   
        df = pd.DataFrame(list_recs, columns=['path', 'label'])
        return df
                
    def clean_db(self):
        """ Reset the 'downloaded' column in the recordings table and update it
        again for all recordings for which slices exist in 'signal_dir'. """
        rec_id_list = list(self.slices_per_downloaded_recording().keys())
        sql_selectors.reset_downloaded(self.conn)
        sql_selectors.set_downloaded(self.conn, rec_id_list)


    def slices_per_species_downloaded(self):
        """ Retrieves Dataframe with class names for currently available slices
        and groups by class """
        df = self.inventory_df().rename(columns={'path':'downloaded_slices'})
        return df.groupby('label').downloaded_slices.count().astype(int).sort_values().to_frame()
        
    def download_missing(self):
        # reimplement
        
        to_download = self.Selection.missing_recordings
        if len(to_download) == 0:
            print('Nothing to download, Selection compltete.')
            return
        
        self._download_threaded(to_download)
        # Log update of slices:
        new_balances = self.slices_per_species_downloaded()
        differences = new_balances - balances
        print(differences[differences > 0])
    
    def _download_threaded(self, recordings):
        # Handle recordings in bunches of 24 to avoid filling tmp too much:
        at_a_time = 24
        total = len(recordings)
        print(f'Downloading {total} recording(s)')
        for iteration, bunch in enumerate([recordings[i:i+at_a_time] for i in range(0, len(recordings), at_a_time)]):
            self.SignalSlicer(bunch)
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(100 * iteration // total)
            bar = fill * filledLength + '-' * (100 - filledLength)
            print('\r%s |%s| %s%% %s' % ('', bar, percent, ''), end = '\r')
            if iteration == total: 
                print()
        
        print('Done downloading!')
        # Update DB:
        rec_ids_to_download = list(map((lambda x: str(x[0])), recordings))
        sql_selectors.set_downloaded(self.conn, rec_ids_to_download)
        return
    
    def seconds_per_species_local_remote(self):
        """ This compares the total seconds of audio material available for each
        species that has already been downloaded vs what is still available.   
        """
        downloaded = sql_selectors.downloaded_german_recordings(self.conn)
        downloaded['downloaded'] = 1
        
        not_downloaded = sql_selectors.not_downloaded_german_recordings(self.conn)
        not_downloaded['downloaded'] = 0
        df = pd.concat([downloaded, not_downloaded])
        return df

    def resample_df(self, df, samples_per_class):
        """ Up- or downsample a dataframe by randomly picking a fixed number of 
        samples for each class """
        out = df.groupby('label').apply(lambda x: x.sample(n=samples_per_class, replace=True)).reset_index(drop=True)
        return out.sample(frac=1).reset_index(drop=True)
    
    def slices_per_downloaded_recording(self):
        files = os.listdir(self.signal_dir)
        rec_ids = [file.split('_')[0] for file in files]
        return {k: v for k,v in zip(Counter(rec_ids).keys(), Counter(rec_ids).values())}
        
    def make_some_noise(self):
        balances = self.slices_per_species_downloaded()
        labels = balances.index.values
            
        # Get noise for all species currently downloaded
        recordings = []
        for label in labels:
            recordings_for_class = sql_selectors.recordings_for_noise(self.conn, label, 1)
            recordings += recordings_for_class
        print(f'Selected {len(recordings)} recordings for noise slicing')
        rec_ids_to_download = list(map((lambda x: str(x[0])), recordings))
        
        # Handle recordings in bunches of 24 to avoid filling tmp too much:
        at_a_time = 24
        for bunch in [recordings[i:i+at_a_time] for i in range(0, len(recordings), at_a_time)]:
            self.NoiseSlicer(bunch)
            urlcleanup()
    
    """        
    def _plot_slices_before_after_downloading(self, balances, differences):
        now = datetime.datetime.now()
        df = pd.DataFrame({'before' : balances, 'added' : differences})
        df.plot(kind='barh', stacked=True)
        plt.savefig(f'Class balances {now}.pdf', bbox_inches = "tight")
    
    def _plot_seconds_per_species_local_remote(self, df):
        now = datetime.datetime.now()
        dl = df[df.downloaded == 1].groupby('label').scraped_duration.sum().sort_values()
        ndl = df[df.downloaded == 0].groupby('label').scraped_duration.sum().sort_values()
        plt.figure(figsize=(50,150))
        p2 = plt.barh(ndl.index.values, ndl)
        p1 = plt.barh(dl.index.values, dl)
        plt.legend((p1[0], p2[0]), ('Downloaded', 'Not Downloaded'))
        plt.savefig(f'Downloaded vs not Downloaded {now}', bbox_inches = "tight")
    """
