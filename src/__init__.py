import Spotify as spotify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.style.use('seaborn')

# Spotify API

spotify = spotify.Spotify()
pink_matter_id = spotify.GetTrackID('Pink Matter')
otra_noche_sin_ti = spotify.GetTrackID('Otra Noche Sin Ti J Balvin')
otra_noche_sin_ti

# Types of plots that would be interesting to look at
#   histogram plot of popularity
#   Scatter plots of popularity vs key components

# Working with the actual data

def ArtistsFormatting(track):
    """format the track artists

    Parameters
    ----------
    track : type
        track data

    Returns
    -------
    list<tuple<string, string>>
        A list of artist id and artist name
    """
    artists = track['artists'].replace("'", '').replace("[", '').split(']')[:-1]
    artists_ids = track['id_artists'].replace("'", '').replace("[", '').split(']')[:-1]

    try:
        return [(artists[idx], artists_ids[idx]) for idx in range(len(artists))]
    except:
        return []

# sample has to be data length and add with replacement
def Bootstrap(data, samples=1000000, size=1000):
    """Bootstrap our data with replacement

    Parameters
    ----------
    data : list
        list of data
    samples : type
        amount of samples per data
    size : type
        size of our array of samples

    Returns
    -------
    ndarray<float>
        means of every sample
    """
    return np.mean([np.random.choice(data, size=samples, replace=True) for _ in range(size)], axis=0)

def GetReleaseYear(date):
    """The year from the date formatted

    Parameters
    ----------
    date : string
        date string object (can vary YYYY, YYYY-MM, YYYY-DD-MM)

    Returns
    -------
    int
        year parsed to int

    """
    return int(date.split("-")[0])

def GetTwoArtists(artist_one, artist_two, years=None, nofeatures=True, metric='popularity', save=False, nameAppend=""):
    """Get the test for two different artists according to a metric

    Parameters
    ----------
    artist_one : list<track>
        a list of analysis track objects
    artist_two : list<track>
        a list of analysis track objects
    years : string
        years to look for in spotify data
    nofeatures : boolean
        do we want data with features
    metric : string
        the metric to look for
    save : boolean
        save figure?
    nameAppend : string
        text to append to filename
    Returns
    -------
    artist_one
        artist_one track analysis
    artist_two
        artist_two track analysis
    """
    labels = [artist_one, artist_two]

    if nofeatures:
        artist_one += '  NOT feat AND NOT with AND NOT ft'
        artist_two += '  NOT feat AND NOT with AND NOT ft'

    if years is not None:
        artist_one += f' year:{years}'
        artist_two += f' year:{years}'

    artist_one = spotify.GetTracksAnalysis(artist_one)
    artist_two = spotify.GetTracksAnalysis(artist_two)

    spotify.Hist(artist_one['analysis'], labels=labels, stacks=[artist_two['analysis']], alpha=0.5)
    spotify.CompareArtistsCLT([artist_one, artist_two], labels=labels, save=save, metric=metric, nameAppend=nameAppend)

    return artist_one, artist_two

tracks_data = pd.read_csv('../data/tracks.csv')
tracks_data['artists'] = tracks_data.apply(ArtistsFormatting, axis=1)
tracks_data['duration'] = tracks_data['duration_ms'].apply(lambda x: x / 60000)
tracks_data['release_year'] = tracks_data['release_date'].apply(GetReleaseYear)
tracks_data.drop('id_artists', axis=1,inplace=True)
tracks_data.drop('duration_ms', axis=1,inplace=True)


spotify.Hist(tracks_data, metrics=['duration'])

spotify.JointPlot(tracks_data, 'danceability', 'release_year')

spotify.Scatter(tracks_data, metrics=['explicit', 'danceability', 'duration'], save=True, sample=0.01)

spotify.BoxPlot(tracks_data, metrics=['danceability'], save=True, nameAppend="danceability", showfliers=False)

spotify.Scatter(tracks_data, metrics=['explicit'])

# <editor-fold> Popularity 0 vs Popularity Non 0

tracks_data_pop_zero = tracks_data[tracks_data['popularity'] == 0].copy()
tracks_data_pop_nozero = tracks_data[tracks_data['popularity'] > 0].copy()
spotify.Hist(tracks_data_pop_nozero, metrics=['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration'], stacks=[tracks_data_pop_zero], labels=['No Pop Zero', 'Pop Zero'], alpha=0.4, save=True, nameAppend="popzero-vs-nonzero")

#</editor-fold>

# <editor-fold> Popularity vs Explicit

# 5866.72
spotify.CatScatter(tracks_data, sample=0.01)

# Get the sum of tracks with no explicitnes and sum up the popularity

sample_mean = np.mean(tracks_data['popularity'])
sample_mean

len(tracks_data[tracks_data['explicit'] == 1]['popularity'])
explicit_pop_mean = np.mean(tracks_data[tracks_data['explicit'] == 1]['popularity'])

len(tracks_data[tracks_data['explicit'] == 0]['popularity'])
non_explicit_pop_mean = np.mean(tracks_data[tracks_data['explicit'] == 0]['popularity'])

# H0: explicit_pop_mean = non_explicit_pop_mean
# HA: explicit_pop_mean > non_explicit_pop_mean

spotify.Hist(tracks=tracks_data[tracks_data['explicit'] == 1], metrics=['popularity'], save=True, nameAppend="explicit-non-norm")
spotify.Hist(tracks=tracks_data[tracks_data['explicit'] == 0], metrics=['popularity'],  save=True, nameAppend="nonexplicit-non-norm")

# non norm calls for bootstrap mean in order to normalize

explicit_norm = Bootstrap(tracks_data[tracks_data['explicit'] == 1]['popularity'], samples=len(tracks_data[tracks_data['explicit'] == 1]['popularity']), size=1000)
nonexplicit_norm = Bootstrap(tracks_data[tracks_data['explicit'] == 0]['popularity'], samples=len(tracks_data[tracks_data['explicit'] == 1]['popularity']), size=1000)

fig, ax = plt.subplots(1, figsize=(7, 7))
ax.hist(explicit_norm, label="Explicit")
ax.hist(nonexplicit_norm, label="Non Explicit")
ax.set_xlabel('Popularity')
ax.set_title('Distribution of Popularity Means')
fig.tight_layout()
fig.legend()
fig.savefig('../plots/bootstrapped-explicit-popularity')
fig

explicit_norm_confidence = stats.t.interval(0.95, len(explicit_norm)-1, loc=np.mean(explicit_norm), scale=stats.sem(explicit_norm))
nonexplicit_norm_confidence = stats.t.interval(0.95, len(nonexplicit_norm)-1, loc=np.mean(nonexplicit_norm), scale=stats.sem(nonexplicit_norm))

explicit_norm_confidence
nonexplicit_norm_confidence

# I will reject the null hypothesis because my confidence intervals do not overlap, there is enough evidence to show the explicit pop mean > non_explicit pop mean

# </editor-fold>

# <editor-fold> Popularity vs Song Length

# H0: Song length <= 5 popularity = song length > 5min
# HA: song length <=5 popularity > song length > 5 min

np.mean(tracks_data[tracks_data['duration'] <= 5]['popularity'])
np.mean(tracks_data[tracks_data['duration'] > 5]['popularity'])

# data is not normally distributed
spotify.Hist(tracks_data[tracks_data['duration'] <= 5], metrics=['popularity'], stacks=[tracks_data[tracks_data['duration'] > 5]], labels=["<= 5min", "> 5min"], alpha=0.4, save=True, nameAppend="durations_lte5_gt5")
spotify.Hist(tracks_data[tracks_data['duration'] > 5], metrics=['popularity'])

duration_ltefive_norm = stats.norm(np.mean(tracks_data[tracks_data['duration'] <= 5]['popularity']), stats.sem(tracks_data[tracks_data['duration'] <= 5]['popularity']))
duration_gtfive_norm = stats.norm(np.mean(tracks_data[tracks_data['duration'] > 5]['popularity']), stats.sem(tracks_data[tracks_data['duration'] > 5]['popularity']))

duration_norm_x = np.linspace(26.7, 27.8, 100000)

fig, ax = plt.subplots(1, figsize=(7, 7))
ax.plot(duration_norm_x, duration_ltefive_norm.pdf(duration_norm_x), label="Duration <= 5")
ax.plot(duration_norm_x, duration_gtfive_norm.pdf(duration_norm_x), label="Duration > 5")
duration_norm_pvalue = round(stats.ttest_ind(tracks_data[tracks_data['duration'] <= 5]['popularity'], tracks_data[tracks_data['duration'] > 5]['popularity'], equal_var=False).pvalue / 2, 3)
ax.text(0.1, 0.8, f'p-value: {(duration_norm_pvalue)}', fontsize=14, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
ax.set_xlabel('Popularity')
ax.set_title('Duration vs Popularity')
fig.tight_layout()
fig.legend()
fig.savefig('../plots/duration_popularity_clt_norm')
fig


# Since the p value is super small, I will reject the null hypothesis, there is enough evidence to show the popularity of songs with length <= 5 min > songs with > 5min

# </editor-fold>

tracks_data_ltfive = tracks_data[tracks_data['duration'] <= 5].copy()

# fix error in data
tracks_data_ltfive.loc[tracks_data_ltfive['id'] == '74CSJTE5QQp1e4bHzm3wti', 'release_year'] = 2019

# <editor-fold> By Years and Metrics

decades = np.arange(1910, 1940, 10)

fig, axs = plt.subplots(len(decades) - 1, figsize=(8, 7 * len(decades) - 1))

for idx, ax in enumerate(axs.flatten()):
    x = tracks_data_ltfive[(tracks_data_ltfive['release_year'] >= decades[idx]) & (tracks_data_ltfive['release_year'] < decades[idx + 1])]['danceability']
    ax.set_title(f"Danceability in {decades[idx]}s mu={x.mean()}")
    ax.hist(x)
    ax.set_xlabel('Danceability')
fig.tight_layout()
fig

# Now we are going to line graph the changes in danceability over the years for funzies

averages = tracks_data_ltfive.groupby('release_year').mean()

averages_norm = averages / tracks_data_ltfive.groupby('release_year').count()

spotify.Years(averages_norm, metrics=['explicit', 'popularity', 'danceability', 'duration'], sameax=False, save=True, nameAppend="plotted-together-years-norm")
spotify.Years(averages, metrics=['explicit', 'popularity', 'danceability', 'duration'], sameax=False, save=True, nameAppend="plotted-together-years")
# </editor-fold>

tracks_data.corr()

# <editor-fold> This years data

tracks_this_year = tracks_data_ltfive[tracks_data_ltfive['release_year'] == 2021].copy()

spotify.Hist(tracks_this_year, metrics=['danceability'], save=True, nameAppend="danceability-2021")

tracks_this_year_mu = tracks_this_year['danceability'].mean()
tracks_this_year_ste = tracks_this_year_mu / np.sqrt(len(tracks_this_year['danceability']))
tracks_this_year_norm = stats.norm(tracks_this_year_mu, tracks_this_year_ste)
tracks_this_year_x = np.linspace(tracks_this_year_mu - (4 * tracks_this_year_ste), tracks_this_year_mu + (4 * tracks_this_year_ste), 1000)

fig, ax = plt.subplots(1, figsize=(7, 7))
ax.plot(tracks_this_year_x, tracks_this_year_norm.pdf(tracks_this_year_x))
ax.set_title('Distribution of Danceability For 2021')
ax.set_xlabel('Danceability')
fig.tight_layout()
fig.savefig('../plots/danceability-2021')
fig

# 95 confidence interval for mean danceability

tracks_this_year_norm.ppf([0.025, 0.975])

# with 95% confidence the dancibility for this years dancibility is between array([0.65710789, 0.69124522])
# would be very very surprising if by the end of the year the dancibility had a mean of 0.71

# </editor-fold>

# Using Spotify API to check the songs of different artists and make hypothesis tests

# Bootstrap in order to check confidence interval
# H0: Drake popularity = G Herbo popularity
# HA: Drake popularity > G Herbo

GetTwoArtists('Kanye West', 'J Cole', metric='popularity', save=True, nofeatures=False)
GetTwoArtists('XXX Tentacion', 'Juice World', metric='popularity', save=True, nofeatures=False)


GetTwoArtists('Bad Bunny', 'J Balvin', metric='danceability', save=True, nofeatures=False)

# using bootstrap


# Notes after seeing the plots compared to popularity
# Maybe some difference
