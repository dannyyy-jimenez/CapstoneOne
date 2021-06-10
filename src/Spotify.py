import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from matplotlib.lines import Line2D
from KEY import key

API_ENDPOINT = 'https://api.spotify.com/v1/'

sns.set_theme(style="ticks", color_codes=True)
plt.style.use('seaborn')


class Spotify():

    def __init__(self):
        access_token = requests.post('https://accounts.spotify.com/api/token', data={'grant_type': 'client_credentials'}, headers={'Authorization': f'Basic {key}', 'Content-type': 'application/x-www-form-urlencoded'}).json()['access_token']
        self.access_token_header = f"Bearer {access_token}"

    def GetTrackID(self, query='Pink Matter'):
        """Return the track id of the foremost query matching spotify track

        Parameters
        ----------
        query : type
            Track name

        Returns
        -------
        string
            The track id

        """
        headers = {'Authorization': self.access_token_header}
        params = {'q': query, 'type': 'track'}
        search = requests.get(API_ENDPOINT + 'search', params=params, headers=headers)
        return search.json()['tracks']['items'][0]['id']

    def GetTracks(self, query='Frank Ocean', type='track', limit=50, offset=0):
        """Return the tracks that match the query

        Parameters
        ----------
        query : type
            Search query

        Returns
        -------
        list<tuple<string, string>>
            A list of tuples that matches the song id and the song name

        """
        headers = {'Authorization': self.access_token_header}
        params = {'q': query, 'type': type, 'limit': limit, 'offset': offset}
        search = requests.get(API_ENDPOINT + 'search', params=params, headers=headers)
        return [(track['id'], track['name']) for track in search.json()['tracks']['items']]

    def GetTrackSpecs(self, id):
        """Return key components of track

        Parameters
        ----------
        id : type
            The Spotify ID of a track

        Returns
        -------
        string
            Track Name
        string
            Track url to listen to it
        array<Tuple>
            ID and Name of every artist in the song
        string
            release date
        int
            Is the track explicit (0 == False, 1 == True)
        float
            track popularity
        int
            amount of available markets track is available in
        """
        headers = {'Authorization': self.access_token_header}
        endpoint = ''.join([API_ENDPOINT, 'tracks/', id])
        track = requests.get(endpoint, headers=headers).json()
        artists = [(artist['id'], artist['name']) for artist in track['artists']]

        return (
            track['name'],
            track['external_urls']['spotify'],
            artists,
            track['album']['release_date'],
            1 if track['explicit'] else 0,
            track['popularity'],
            len(track['available_markets'])
        )

    def GetTrackAudioFeatures(self, id):
        """Get track audio features from the spotify api

        Parameters
        ----------
        id : type
            Spotify ID of the song

        Returns
        -------
        (AudioFeaturesObjects) - https://developer.spotify.com/documentation/web-api/reference/#object-audiofeaturesobject

        float
            A value of 0.0 is least danceable and 1.0 is most danceable.
        float
            Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
        int
            The key the track is in. E.g. 0 = C, 1 = C♯/D♭, 2 = D
        float
            The overall loudness of a track in decibels (dB). Values typical range between -60 and 0 db.
        int
            Mode indicates the modality (major or minor). Major is represented by 1 and minor is 0.
        float
            Speechiness detects the presence of spoken words in a track.
                Values above 0.66 describe tracks that are probably made entirely of spoken words.
                Values between 0.33 and 0.66 describe tracks that may contain both music and speech.
                Values below 0.33 most likely represent music and other non-speech-like tracks.
        float
            A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
        float
            Predicts whether a track contains no vocals. confidence is higher as the value approaches 1.0
        float
            Detects the presence of an audience in the recording.
        float
            A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track
        float
            The overall estimated tempo of a track in beats per minute (BPM).
        int
            The duration of the track in milliseconds.
        int
            An estimated overall time signature of a track. how many beats are in each bar (or measure).
        """
        headers = {'Authorization': self.access_token_header}
        endpoint = ''.join([API_ENDPOINT, 'audio-features/', id])
        audio_features = requests.get(endpoint, headers=headers).json()

        return (
            audio_features['danceability'],
            audio_features['energy'],
            audio_features['key'],
            audio_features['loudness'],
            audio_features['mode'],
            audio_features['speechiness'],
            audio_features['acousticness'],
            audio_features['instrumentalness'],
            audio_features['liveness'],
            audio_features['valence'],
            audio_features['tempo'],
            round(audio_features['duration_ms'] / 60000, 2),
            audio_features['time_signature']
        )

    def GetTrackAnalysis(self, id):
        """Get all info on a track, specs and audio features in a pretty format

        Parameters
        ----------
        id : type
            Spotify ID of a track

        Returns
        -------
        dict
            Pretty track info formatted

        """
        specs = self.GetTrackSpecs(id)
        audio_features = self.GetTrackAudioFeatures(id)

        return {
            'name': specs[0],
            'url': specs[1],
            'artists': specs[2],
            'release_data': specs[3],
            'explicit': specs[4],
            'popularity': specs[5],
            'availability': specs[6],
            'danceability': audio_features[0],
            'energy': audio_features[1],
            'key': audio_features[2],
            'loudness': audio_features[3],
            'mode': audio_features[4],
            'speechiness': audio_features[5],
            'acousticness': audio_features[6],
            'instrumentalness': audio_features[7],
            'liveness': audio_features[8],
            'valence': audio_features[9],
            'tempo': audio_features[10],
            'duration': audio_features[11],
            'time_signature': audio_features[12]
        }

    def GetTracksAnalysis(self, query='Drake NOT feat AND NOT with AND NOT ft'):
        """Get tracks analysis for multiple tracks

        Parameters
        ----------
        query : type
            Search query for Spotify API

        Returns
        -------
        dict
            tracks
                the tracks object
            analysis
                the tracks analysis list of objects
        """
        artist_tracks = self.GetTracks(query)
        artist_tracks_analysis = [self.GetTrackAnalysis(track[0]) for track in artist_tracks]
        return {'tracks': artist_tracks, 'analysis': pd.DataFrame(artist_tracks_analysis)}

    def FormatMetric(self, metric):
        """Format metrics into the capitalized version of them

        Parameters
        ----------
        metric : type
            dirty version of metric

        Returns
        -------
        string
            formatted version of metric
        """
        if metric == 'duration':
            return 'Duration (min)'
        return ' '.join([metric.capitalize() for metric in metric.split("_")])

    def Scatter(self, tracks, dependent='popularity', metrics=['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration'], labelize=False, save=False, nameAppend=""):
        """Create various scatter plots for each metric

        Parameters
        ----------
        tracks : list<tracks>
            An array of track analysis objects
        dependent : string
            the y axis metric
        metrics : list<string>
            metrics to measure on our x axis
        labelize : boolean
            Whether to include legend
        save : boolean
            save figuire to machine
        nameAppend : string
            text to append to the filename

        Returns
        -------
        None
        """
        if isinstance(tracks, list):
            tracks = pd.DataFrame(tracks)

        fig, axs = plt.subplots(len(metrics), figsize=(10, 6*len(metrics)))

        colors = np.random.rand(len(tracks), 3)
        labels = tracks['name'].values
        y = tracks[dependent].values
        legend_elements = []
        tracks_enum = enumerate([axs]) if len(metrics) == 1 else enumerate(axs.flatten())

        for i in range(len(labels)):
            legend_elements.append(Line2D([0], [0], marker="o", markersize=3, color=colors[i], label=labels[i], linestyle=None))

        for idx, ax in tracks_enum:
            if labelize:
                ax.scatter(x=tracks[metrics[idx]], y=y, c=colors)
                # for i in range(len(labels)):
                #     ax.annotate(labels[i], (tracks[metrics[idx]][i], y[i]))
            else:
                ax.scatter(x=tracks[metrics[idx]], y=y, alpha=0.1)

            ax.set_ylabel(self.FormatMetric(dependent))
            ax.set_xlabel(self.FormatMetric(metrics[idx]))
            ax.set_title(f'{self.FormatMetric(dependent)} vs {self.FormatMetric(metrics[idx])}')

        if labelize:
            fig.legend(handles=legend_elements, loc=1)
        fig.tight_layout()

        if save:
            filename = f"../plots/popularity-vs-{'_'.join(metrics)}-{nameAppend}"
            fig.savefig(filename)

    def Hist(self, tracks, metrics=['popularity'], save=False, nameAppend="", labels=None, stacks=[], alpha=1):
        """Plot histograms for track analysis objects

        Parameters
        ----------
        tracks : list<track>
            A list of track analysis objects
        metrics : list<string>
            list of metrics to plot on our x axis
        save : boolean
            should the figure be saved?
        nameAppend : string
            text to append to filename
        labels : list<string>
            list of labels to give the histograms
        stacks : list<tracks>
            list of track analysis objects to stack on top of the axis
        alpha : float (0, 1]
            alpha value of every histogram

        Returns
        -------
        None
        """
        if isinstance(tracks, list):
            tracks = pd.DataFrame(tracks)

        fig, axs = plt.subplots(len(metrics), figsize=(20, 6*len(metrics)))

        tracks_enum = enumerate([axs]) if len(metrics) == 1 else enumerate(axs.flatten())

        for idx, ax in tracks_enum:
            if labels is not None and idx == 0:
                ax.hist(x=tracks[metrics[idx]], label=labels[0], alpha=alpha)
            else:
                ax.hist(x=tracks[metrics[idx]], alpha=alpha)

            for idxs, stack in enumerate(stacks):
                if isinstance(stack, list):
                    stack = pd.DataFrame(stack)

                if idx == 0:
                    ax.hist(x=stack[metrics[idx]], label=labels[idxs+1], alpha=alpha)
                else:
                    ax.hist(x=stack[metrics[idx]], alpha=alpha)

            ax.set_xlabel(self.FormatMetric(metrics[idx]))
            ax.set_title(f'Distribution of {self.FormatMetric(metrics[idx])}')

        if labels is not None:
            fig.legend()
        fig.tight_layout()

        if save:
            filename = f"../plots/popularity-hist-{nameAppend}"
            fig.savefig(filename)

    def CatScatter(self, tracks, x='explicit', y='popularity', sample=0.2):
        """Make a categorical scatter plot

        Parameters
        ----------
        tracks : list<tracks>
            List object for tracks to compare
        x : string
            metric to use for x axis
        y : string
            metric to use for y axis
        sample : float [0, 1]
        Returns
        -------
        None
        """
        sample_loc = np.random.choice(np.arange(len(tracks)), size=int(np.ceil(len(tracks) * sample)), replace=False)
        tracks_sample = tracks.iloc[sample_loc].copy()

        if isinstance(tracks, list):
            tracks = pd.DataFrame(tracks)
        catplot = sns.catplot(x=x, y=y, data=tracks_sample, jitter=0.4, height=8, kind="violin")
        catplot.set_axis_labels(self.FormatMetric(x), self.FormatMetric(y))
        catplot.set_xticklabels(["No", "Yes"])
        catplot.set_titles(col_template="Distribution of Popularity Based on Explicitness")
        catplot.fig.suptitle('Distribution of Popularity Based on Explicitness', fontsize=16)
        catplot.tight_layout()
        catplot.fig.savefig('../plots/cat-scatter-explicit-pop')

    def Years(self, tracks, x='release_year', metrics=['explicit', 'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration'], sameax=False, save=False, nameAppend=""):
        """Short summary.

        Parameters
        ----------
        tracks : list<tracks>
            list of track analysis objects
        x : string
            x axis metric to plot
        metrics : list<string>
            metrics to plot over the years
        sameax : boolean
            use the same axis
        save : boolean
            save figure?
        nameAppend : string
            text file name appending text

        Returns
        -------
        None
        """
        if isinstance(tracks, list):
            tracks = pd.DataFrame(tracks)

        if sameax:
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.set_title("Change in Metrics Over Time")
            for idx, metric in enumerate(metrics):
                ax.plot(tracks.index, tracks[metrics[idx]].values, label=self.FormatMetric(metrics[idx]))
            fig.legend()
        else:
            fig, axs = plt.subplots(len(metrics), figsize=(8, 8 * len(metrics)))
            for idx, ax in enumerate(axs.flatten()):
                ax.set_title(f"Change in {self.FormatMetric(metrics[idx])} Over Time")
                ax.set_ylabel(self.FormatMetric(metrics[idx]))
                ax.plot(tracks.index, tracks[metrics[idx]].values)

        fig.tight_layout()
        if save:
            filename = f"../plots/years-{nameAppend}"
            fig.savefig(filename)

    def CompareArtistsCLT(self, artists, metric='popularity', labels=[], save=False, nameAppend=""):
        """Compare artists metrics using central limit theorem and t testing

        Parameters
        ----------
        artists : list<track data>
            artists data to use
        metric : string
            metric to measure
        labels : list<string>
            list of strings to label our normal models
        save : boolean
            save figure?
        nameAppend : string
            text to append to data

        Returns
        -------
        float
            p value  of our t test
        """
        artist_one_tracks_pop_mean = np.mean(artists[0]['analysis'][metric])
        artist_one_tracks_pop_ste = stats.sem(artists[0]['analysis'][metric])

        artist_two_tracks_pop_mean = np.mean(artists[1]['analysis'][metric])
        artist_two_tracks_pop_ste = stats.sem(artists[1]['analysis'][metric])

        artist_one_tracks_norm = stats.norm(artist_one_tracks_pop_mean, artist_one_tracks_pop_ste)
        artist_two_tracks_norm = stats.norm(artist_two_tracks_pop_mean, artist_two_tracks_pop_ste)

        if artist_one_tracks_pop_mean < artist_two_tracks_pop_mean:
            artists_norm_x = np.linspace(artist_one_tracks_pop_mean - (4 * artist_one_tracks_pop_ste), artist_two_tracks_pop_mean + (4 * artist_two_tracks_pop_ste), 100000)
        else:
            artists_norm_x = np.linspace(artist_two_tracks_pop_mean - (4 * artist_two_tracks_pop_ste), artist_one_tracks_pop_mean + (4 * artist_one_tracks_pop_ste), 100000)

        pvalue = (stats.ttest_ind(artists[0]['analysis'][metric], artists[1]['analysis'][metric], equal_var=False).pvalue) / 2

        fig, ax = plt.subplots(1, figsize=(7, 7))
        ax.text(0.1, 0.8, f'p-value: {pvalue}', fontsize=14, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        ax.set_xlabel(f"Mean {self.FormatMetric(metric)} Distribution")
        ax.set_title(f"{labels[0]} vs {labels[1]}")
        ax.plot(artists_norm_x, artist_one_tracks_norm.pdf(artists_norm_x), label=labels[0])
        ax.plot(artists_norm_x, artist_two_tracks_norm.pdf(artists_norm_x), label=labels[1])
        fig.legend()
        fig.tight_layout()
        if save:
            filename = f"../plots/comparison-testing-{metric}-{labels[0]}>{labels[1]}-{nameAppend}"
            fig.savefig(filename)

        return pvalue


if __name__ == "__main__":
    spotify = Spotify()
    # pink_matter_id = spotify.GetTrackID('Pink Matter')
    # a_tu_merced_id = spotify.GetTrackID('A Tu Merced Bad Bunny')
    #
    # pink_matter = spotify.GetTrackAnalysis(pink_matter_id)
    # a_tu_merced = spotify.GetTrackAnalysis(a_tu_merced_id)

    steezy_la_flame = spotify.GetTracksAnalysis('Bad Bunny')
    benji = spotify.GetTracksAnalysis('Mike Towers')

    spotify.CompareArtistsCLT([benji, steezy_la_flame], labels=['6enji', 'SteezyLaFlame'])
