import os
import urllib
import urllib2
import BeautifulSoup
import urllib2


if __name__ == '__main__':
    rfc_url = "http://www.cv.nrao.edu/2cmVLBA/data/multifreq/"
    download_dir = '/home/ilya/code/vlbi_errors/examples/'
    request = urllib2.Request(rfc_url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup.BeautifulSoup(response)

    source = '1226+023'
    bands = ['x', 'y', 'j', 'y']
    download_list = list()
    for a in soup.findAll('a'):
        if source in a['href'] and '.uvf' in a['href']:
            fname = a['href']
            download_list.append(os.path.join(rfc_url, fname))

    for url in download_list:
        fname = os.path.split(url)[-1]
        print("Downloading file {}".format(fname))
        urllib.urlretrieve(url, os.path.join(download_dir, fname))
