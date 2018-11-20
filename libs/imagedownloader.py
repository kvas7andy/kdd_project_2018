import sys
import ssl
import socket
import os
import time
import tarfile

import pandas

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse
    import urllib

class ImageNetDownloader:
    def __init__(self):
        self.host = 'http://www.image-net.org'

    def download_file(self, url, desc=None, renamed_file=None):
        socket.setdefaulttimeout(1.0)
        try:
            if 'img2.pcauto.com' in url or 'img.315che.com' in url:
                raise MyException("There was an error: with url domain")
            u = urllib2.urlopen(url, timeout=2)
        except (socket.timeout, urllib2.URLError) as e:
            # For Python 2.7
            raise MyException("There was an error: %r" % e)

        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        filename = os.path.basename(path)
        if not filename:
            filename = 'downloaded.file'

        if not renamed_file is None:
            filename = renamed_file

        if desc:
            filename = os.path.join(desc, filename)

        if os.path.exists(filename):
            return filename


        with open(filename, 'wb') as f:
            meta = u.info()
            meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
            meta_length = meta_func("Content-Length")
            meta_type = meta_func("Content-Type")

            file_size = None

            if meta_type[0] != "image/jpeg":
                print "Type is ", meta_type
                f.close()
                return None
            if meta_length:
                file_size = int(meta_length[0])
                if file_size <= 2*2**3*10**3:
                    print "Size is ", file_size
                    f.close()
                    return None
            print "Downloading: {0} Bytes: {1} \t Name: {2}".format(url, file_size, filename.rsplit('/')[-1])

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)

                status = "{0:16}".format(file_size_dl)
                if file_size:
                    status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
                status += chr(13)
            if file_size_dl < 2*2**3*2**10: # 2kB
                print file_size_dl, filename.rsplit('/')[-1]
                f.close()
                return None
                #f.close()
                #os.remove(f.name)

        return filename

    def extractTarfile(self, filename):
        tar = tarfile.open(filename)
        tar.extractall()
        tar.close()

    def downloadBBox(self, wnid):
        filename = str(wnid) + '.tar.gz'
        url = self.host + '/downloads/bbox/bbox/' + filename
        try:
            filename = self.download_file(url, self.mkWnidDir(wnid))
            currentDir = os.getcwd()
            os.chdir(wnid)
            self.extractTarfile(filename)
            print 'Download bbbox annotation from ' + url + ' to ' + filename
            os.chdir(currentDir)
        except Exception, error:
            print 'Fail to download' + url

    def getImageURLsOfWnid(self, wnid):
        url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + str(wnid)
        while True:
            try:
                f = urllib.urlopen(url)
                contents = f.read().split('\n')
                break
            except IOError as e:
                print e.__class__.__name__

        imageUrls = []

        for each_line in contents:
            # Remove unnecessary char
            each_line = each_line.replace('\r', '').strip()
            if each_line:
                imageUrls.append(each_line)

        return imageUrls

    def mkWnidDir(self, wnid):
        if not os.path.exists(wnid):
            os.mkdir(wnid)
        return os.path.abspath(wnid)

    def downloadImagesByURLs(self, wnid, imageUrls, filename_info=None):
        # save to the dir e.g: n005555_urlimages/
        urlimages_dir = os.path.join(filename_info['dataset_dir'], 'train')#, self.mkWnidDir(wnid), str(wnid) + '_urlimages')
        if not os.path.exists(urlimages_dir):
            os.makedirs(urlimages_dir)

        for url in imageUrls:
            try:
                status = self.download_file(url, urlimages_dir, renamed_file=filename_info['filename'])
                #print status

                if status is None:
                    #print filename_info['filename']
                    #if os.path.exists(filename_info['filename']):
                    #if os.path.exists(filename_info['filename']):
                    while True:
                        try:
                            print "Remove file ", filename_info['filename']
                            os.remove(os.path.join(urlimages_dir, filename_info['filename']))
                            print "Removed file ", filename_info['filename']
                            break
                        except:
                            time.sleep(1)
                return status

            except (Exception, socket.timeout) as e: #(urllib2.HTTPError, urllib2.URLError, ssl.CertificateError)
                print 'Fail to download : ' + url + '  File: ' + filename_info['filename'] + ' with error ' + e.__class__.__name__
                # if e.__class__.__name__ != 'URLError':
                #     print e.code
                #     print e.msg
                return None

    def downloadOriginalImages(self, wnid, username, accesskey):
        download_url = 'http://www.image-net.org/download/synset?wnid=%s&username=%s&accesskey=%s&release=latest&src=stanford' % (wnid, username, accesskey)
        try:
             download_file = self.download_file(download_url, self.mkWnidDir(wnid), wnid + '_original_images.tar')
        except Exception, erro:
            print 'Fail to download : ' + download_url

        currentDir = os.getcwd()
        extracted_folder = os.path.join(wnid, wnid + '_original_images')
        if not os.path.exists(extracted_folder):
            os.mkdir(extracted_folder)
        os.chdir(extracted_folder)
        self.extractTarfile(download_file)
        os.chdir(currentDir)
        print 'Extract images to ' + extracted_folder