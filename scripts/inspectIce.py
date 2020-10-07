#!/usr/bin/env python
import numpy as np
import scipy
import scipy.ndimage
import h5py as h
import glob
import os, re, sys
import matplotlib
import matplotlib.patches
import photutils
from photutils import centroid_com
import cv2
from scipy.ndimage.morphology import binary_fill_holes
import pylab as plt
from myModules import extractDetectorDist as eDD
from optparse import OptionParser
import warnings
# suppress warnings, see: https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = OptionParser()
parser.add_option("-r", "--run", action="store", type="string", dest="runNumber", help="run number you wish to view", metavar="rxxxx")
parser.add_option("-i", "--inspectOnly", action="store_true", dest="inspectOnly", help="inspect output directory", default=False)
parser.add_option("-o", "--outputDir", action="store", type="string", dest="outputDir", help="output directory will be appended by run number (default: output_rxxxx); separate types will be stored in output_rxxxx/type[1-3]", default="output")
parser.add_option("-t", "--iceType", action="store", type="int", dest="iceType", help="default:1 = output_rxxxx/type1", default=1)
parser.add_option("-m", "--mask", action="store", type="string", dest="maskFile", help="mask file in the output directory to be used for viewing (default: None)", default=None)
parser.add_option("-W", "--waterAveraging", action="store_true", dest="averageWaterTypes", help="average pattern and angavg of water types", default=False)
parser.add_option("-M", "--maxIntens", action="store", type="int", dest="maxIntens", help="doesn't plot intensities above this value (default:2000)", default=2000)
parser.add_option("-S", "--sortTypes", action="store", type="int", dest="sortTypes", help="default:0. -1(descending total intens), 0(peakyness), 1(ascending total intens).", default=0)
parser.add_option("-T", "--thresholdIce", action="store", type="float", dest="thresholdIce", help="sets number of standard deviations and average intensities to use as threshold for peak finding (default:2)", default=2)
parser.add_option("-D", "--peakDimension", action="store", type="float", dest="peakDimension", help="sets threshold for minimum number of pixels for each peak dimension (default:10 pixels)", default=10)
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="print more stuff to terminal", default=False)
(options, args) = parser.parse_args()

#Tagging directories with the correct names
runtag = "r%s"%(options.runNumber)
write_dir = options.outputDir + '_' + runtag + '/'
write_anomaly_dir = write_dir
source_dir = write_dir
if(not os.path.exists(write_anomaly_dir)):
	os.mkdir(write_anomaly_dir)
numTypes = 5
write_anomaly_dir_types = [write_dir]
for i in range(numTypes):
	write_anomaly_dir_types.append(write_anomaly_dir+"type"+str(i+1)+"/") 
ice_dir = [s for s in write_anomaly_dir_types if ("type%d" % 1) in s]
if len(ice_dir) != 1:
	if len(ice_dir) == 0:
		print ("found no folder that matches ice type%d, aborting..")
		sys.exit(-1)
	else:
		print ("found multiple folders that matches ice type%d (only first one will be used):", ice_dir)
ice_dir = ice_dir[0]

if options.maskFile is not None:
	print ("reading mask from: %s .." % options.maskFile)
	f = h.File(write_dir + options.maskFile, 'r')
	mask = (np.array(f['data']['diffraction']) > 0).astype(np.int)
	f.close()
else:
	mask = None


	#Function to detect ice peaks and centroids
def detect_peaks(img, int_threshold, peak_threshold, mask=None, center=None):
	threshold = int_threshold*(np.mean(img)+np.std(img))
	image_thresholded = np.copy(img)
	# TODO: look at only non-masked pixels
	image_thresholded[img<threshold] = 0
	#find the peak regions and label all the pixels
	labeled_image, number_of_peaks = scipy.ndimage.label(image_thresholded)
	peak_regions = scipy.ndimage.find_objects(labeled_image)
	peak_list = []
	for peak_region_i in peak_regions:
		ry = img[peak_region_i].shape[0]
		rx = img[peak_region_i].shape[1]
		if (ry>peak_threshold) and (rx>peak_threshold):
			img_peak = np.zeros_like(img)
			img_peak[peak_region_i] = img[peak_region_i]
			# TODO: improve peak width with 1D Gaussian fit
			if (ry > rx):
				r = ry/2.
			else:
				r = rx/2.
			# current reference position is first element of 2D array (lower left corner), not center
			cy,cx = centroid_com(img_peak)
			#cy -= (img_peak.shape[0]-1)/2.
			#cx -= (img_peak.shape[1]-1)/2.
			peak_list.append([img[peak_region_i],(cy,cx),r])
	return peak_list


def peak_sphericity(img,int_threshold,peak_threshold, BG_level = 100, photon_threshold = 2000):
    p = []
    sphericity_of_peak = []
    # make a list of the bounding box of the peaks
    for i in np.arange(len(detect_peaks(img, int_threshold, peak_threshold))):
        p.append(detect_peaks(img, int_threshold, peak_threshold)[i][0])
    for p_i in np.arange(len(p)):
        p_sub = p[p_i] - BG_level
        p_mask = p_sub > photon_threshold
        fill_holes = binary_fill_holes(p_mask)
        contours,_= cv2.findContours(fill_holes.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if perimeter != 0:
                sphericity = 4*np.pi*area/(perimeter**2)     
                cnt_string = 'Numbers of contours for peak {} : {}, sphericity: {}'
                print(cnt_string.format(p_i, len(contours), sphericity))
                sphericity_of_peak.append((p[p_i],sphericity))
    return sphericity_of_peak


#Change into data directory to extract *angavg.h5 files from the ice anomaly type
arr = []
originaldir=os.getcwd()
os.chdir(ice_dir)
files = glob.glob("LCLS*angavg.h5")
print ("reading ang_avgs from: %s .." % ice_dir)
for i in files:
	f = h.File(i, 'r')
	arr.append(np.array(f['data']['data'][1]))
	f.close()
os.chdir(originaldir)
masterArr = np.array(arr)
numData = len(masterArr)
angAvgLen = len(masterArr[0])

#Normalize to water ring
normed_arr = np.zeros((numData, angAvgLen))
sorted_arr = np.zeros((numData, angAvgLen))
sortedFileNames = []
unnormed_arr = masterArr.copy()
for i in range(numData):
	temp = masterArr[i]
	max_temp = np.max(temp[530:560])
	min_temp = np.min(temp[50:1153])
	normed_arr[i] = (temp - min_temp) / (max_temp - min_temp)

#Sorting routines
if(options.sortTypes==-1):
	print ("sorting by total intensities in descending order..")
	scoreKeeper = [np.sum(np.abs(i)) for i in unnormed_arr]
	ordering = (np.argsort(scoreKeeper))[-1::-1]
	sorted_arr = normed_arr[ordering]
	sortedFileNames = np.array(files)[ordering]
elif (options.sortTypes==1):
	print ("sorting by total intensities in ascending order..")
	scoreKeeper = [np.sum(np.abs(i)) for i in unnormed_arr]
	ordering = np.argsort(scoreKeeper)
	sorted_arr = normed_arr[ordering]
	sortedFileNames = np.array(files)[ordering]
elif (options.sortTypes==0):
	print ("sorting by maximum of median filtered ang_avgs..")
	filterLen = 5
	medianFiltered_arr = np.zeros((numData, angAvgLen-filterLen))
	for i in range(numData):
		for j in range(len(normed_arr[i])-filterLen):
			medianFiltered_arr[i][j] = np.median(normed_arr[i][j:j+filterLen])
	scoreKeeper = [np.max(np.abs(i[201:1001]-i[200:1000])) for i in medianFiltered_arr]
	ordering = np.argsort(scoreKeeper)
	sorted_arr = normed_arr[ordering]
	sortedFileNames = np.array(files)[ordering]

#Global parameters
colmax = options.maxIntens
colmin = 0
storeFlag = 0
########################################################
# Imaging class copied from Ingrid Ofte's pyana_misc code
########################################################
class img_class (object):
	def __init__(self, inarr, inangavg, filename, peakList=None, meanWaveLengthInAngs=eDD.nominalWavelengthInAngs, detectorDistance=eDD.get_detector_dist_in_meters(runtag)):
		self.inarr = inarr*(inarr>0)
		self.filename = filename
		self.inangavg = inangavg
		self.inpeaks = peakList
		self.wavelength = meanWaveLengthInAngs
		self.detectorDistance = detectorDistance
		self.HIceQ ={}
		global colmax
		global colmin
		global storeFlag
		self.tag = 0

	def on_keypress_for_tagging(self,event):
		global colmax
		global colmin
		global storeFlag
		if event.key in [str(i) for i in range(numTypes+1)]:
			storeFlag = int(event.key)
			
			if(options.inspectOnly):
				print ("Inspection only mode.")
			else:
				if(not os.path.exists(write_anomaly_dir_types[storeFlag])):
					os.mkdir(write_anomaly_dir_types[storeFlag])
				pngtag = write_anomaly_dir_types[storeFlag] + "%s.png" % (self.filename)
				if(self.tag != 0):
					#delete previous assignment
					pngtag = write_anomaly_dir_types[self.tag] + "%s.png" % (self.filename)
					if os.path.isfile(pngtag):
						os.remove(pngtag)
						print ("%s removed!" % (pngtag))
					else:
						print ("No action taken.")
					#Save new assignment if it's store flag not type 0
					if (storeFlag !=0):
							pngtag = write_anomaly_dir_types[storeFlag] + "%s.png" % (self.filename)
							plt.savefig(pngtag)
							print ("%s saved." % (pngtag))
							self.tag = storeFlag
				else:
					plt.savefig(pngtag)
					print ("%s saved." % (pngtag))
					self.tag = storeFlag
		if event.key == 'r':
			colmin = self.inarr.min()
			colmax = ((self.inarr<options.maxIntens)*self.inarr).max()
			plt.clim(colmin, colmax)
			plt.draw()

	def on_keypress_for_viewing(self,event):
		global colmax
		global colmin
		global storeFlag
		if event.key == 'p':
			pngtag = write_anomaly_dir_types[storeFlag] + "%s.png" % (self.filename)
			if(options.inspectOnly):
				print ("Inspection only mode.")
			else:
				plt.savefig(pngtag)
				print ("%s saved." % (pngtag))
		if event.key == 'r':
			colmin = self.inarr.min()
			colmax = ((self.inarr<options.maxIntens)*self.inarr).max()
			plt.clim(colmin, colmax)
			plt.draw()

	def on_click(self, event):
		global colmax
		global colmin
		if event.inaxes:
			lims = self.axes.get_clim()
			colmin = lims[0]
			colmax = lims[1]
			range = colmax - colmin
			value = colmin + event.ydata * range
			if event.button is 1 :
				if value > colmin and value < colmax :
					colmin = value
			elif event.button is 2 :
				colmin = self.inarr.min()
				colmax = self.inarr.max()
			elif event.button is 3 :
				if value > colmin and value < colmax:
					colmax = value
			plt.clim(colmin, colmax)
			plt.draw()


	def draw_img_for_viewing(self):
		if(options.verbose and not options.inspectOnly):
			print ("Press 'p' to save PNG.")
		global colmax
		global colmin
		fig = plt.figure(num=None, figsize=(13.5, 5), dpi=100, facecolor='w', edgecolor='k')
		cid1 = fig.canvas.mpl_connect('key_press_event', self.on_keypress_for_viewing)
		cid2 = fig.canvas.mpl_connect('button_press_event', self.on_click)
		canvas = fig.add_subplot(121)
		canvas.set_title(self.filename)
		self.axes = plt.imshow(self.inarr, origin='lower', interpolation='nearest', vmax=colmax, vmin=colmin, cmap='gnuplot')
		self.colbar = plt.colorbar(self.axes, pad=0.01)
		self.orglims = self.axes.get_clim()
		cmap = matplotlib.cm.gnuplot
		cmap.set_bad('grey',1.)
		#cmap.set_under('white',1.)
		canvas = fig.add_subplot(122)
		canvas.set_title("angular average")
		maxAngAvg = (self.inangavg).max()
		for i,j in eDD.iceHInvAngQ.iteritems():
			self.HIceQ[i] = eDD.get_pix_from_invAngsQ_and_detectorDist(runtag,j,self.detectorDistance, wavelengthInAngs=self.wavelength)

		numQLabels = len(self.HIceQ.keys())+1
		labelPosition = maxAngAvg/numQLabels
		for i,j in self.HIceQ.iteritems():
			plt.axvline(j,0,colmax,color='r')
			plt.text(j,labelPosition,str(i), rotation="45")
			labelPosition += maxAngAvg/numQLabels

		plt.plot(self.inangavg)
		plt.show()

	def draw_img_for_tagging(self):
		if(not options.inspectOnly):
			print ("Press 1-"+ str(numTypes)+ " to save png (overwrites old PNGs); Press 0 to undo (deletes png if wrongly saved).")
		global colmax
		global colmin
		global storeFlag
		fig = plt.figure(num=None, figsize=(13.5, 5), dpi=100, facecolor='w', edgecolor='k')
		cid1 = fig.canvas.mpl_connect('key_press_event', self.on_keypress_for_tagging)
		cid2 = fig.canvas.mpl_connect('button_press_event', self.on_click)
		canvas = fig.add_subplot(121)
		canvas.set_title(self.filename)
		self.axes = plt.imshow(self.inarr, origin='lower', interpolation='nearest', vmax=colmax, vmin=colmin, cmap='gnuplot')
		self.colbar = plt.colorbar(self.axes, pad=0.01)
		self.orglims = self.axes.get_clim()
		cmap = matplotlib.cm.gnuplot
		cmap.set_bad('grey',1.)
		#cmap.set_under('white',1.)
		if self.inpeaks is not None:
			# plot peaks (peak[0]: 2D img, peak[1]: center of mass, peak[2]: radius)
			ax = fig.gca()
			for peak in self.inpeaks:
				circ = plt.Circle(peak[1], radius=peak[2], linewidth=2, color='w')
				circ.set_fill(False)
				ax.add_patch(circ)
		canvas = fig.add_subplot(122)
		canvas.set_title("angular average")
		maxAngAvg = (self.inangavg).max()
		for i,j in eDD.iceHInvAngQ.iteritems():
			self.HIceQ[i] = eDD.get_pix_from_invAngsQ_and_detectorDist(runtag,j,self.detectorDistance, wavelengthInAngs=self.wavelength)
		
		numQLabels = len(self.HIceQ.keys())+1
		labelPosition = maxAngAvg/numQLabels
		for i,j in self.HIceQ.iteritems():
			plt.axvline(j,0,colmax,color='r')
			plt.text(j,labelPosition,str(i), rotation="45")
			labelPosition += maxAngAvg/numQLabels
		
		plt.plot(self.inangavg)
		plt.show()
	
	def draw_spectrum(self):
		if options.verbose:
			print ("Press 'p' to save PNG.")
		global colmax
		global colmin
		localColMax=self.inarr.max()
		localColMin=self.inarr.min()
		aspectratio = 1.5*(self.inarr.shape[1])/(float(self.inarr.shape[0]))
		fig = plt.figure(num=None, figsize=(13, 10), dpi=100, facecolor='w', edgecolor='k')
		cid1 = fig.canvas.mpl_connect('key_press_event', self.on_keypress_for_viewing) 
		cid2 = fig.canvas.mpl_connect('button_press_event', self.on_click)
		canvas = fig.add_axes([0.05,0.05,0.6,0.9], xlabel="q", ylabel="normalized angular average")
		canvas.set_title(self.filename)
		self.axes = plt.imshow(self.inarr, origin='lower', aspect=aspectratio, interpolation='nearest', vmax = localColMax, vmin = localColMin)
		self.colbar = plt.colorbar(self.axes, pad=0.01)
		self.orglims = self.axes.get_clim() 
		canvas2 = fig.add_axes([0.7,0.05,0.25,0.9], xlabel="log(sorting score)", ylabel="data")
		canvas2.set_ylim([0,numData])
		canvas2.plot(np.log(np.array(scoreKeeper)[ordering]),range(numData))
		plt.show()  	

if options.verbose:
	print ("Right-click on colorbar to set maximum scale.")
	print ("Left-click on colorbar to set minimum scale.")
	print ("Center-click on colorbar (or press 'r') to reset color scale.")
	print ("Interactive controls for zooming at the bottom of figure screen (zooming..etc).")
	print ("Hit Ctl-\ or close all windows (Alt-F4) to terminate viewing program.")

currImg = img_class(sorted_arr, None, runtag+"_spectrum_sort%s"%(options.sortTypes))
#currImg = img_class(unnormed_arr[ordering], None, runtag+"_spectrum_sort%s"%(options.sortTypes))
currImg.draw_spectrum()
#currImg = img_class(sorted_arr, unnormed_arr.mean(axis=0), runtag+"_spectrum_sort%s"%(options.sortTypes))
#currImg.draw_img_for_viewing()

########################################################
# Loop to display all non-anomalous H5 files. 
########################################################

avgArr = np.zeros((numTypes+1,1760,1760))
avgRadAvg = np.zeros((numTypes+1,1233))
typeOccurences = np.zeros(numTypes+1)
waveLengths={}
for i in range(numTypes):
	waveLengths[i] = []

########################################################
# Loop to display all H5 files with ice anomalies. 
########################################################
if options.verbose:
	print ("Right-click on colorbar to set maximum scale.")
	print ("Left-click on colorbar to set minimum scale.")
	print ("Center-click on colorbar (or press 'r') to reset color scale.")
	print ("Interactive controls for zooming at the bottom of figure screen (zooming..etc).")
	print ("Hit Ctl-\ or close all windows (Alt-F4) to terminate viewing program.")

anomalies = sortedFileNames

waveLengths={}
rangeNumTypes = range(1,numTypes+1)
for i in range(numTypes):
	waveLengths[i] = []

#Tag anomalies
for fname in anomalies:
	storeFlag=0
	diffractionName = ice_dir + re.sub("-angavg",'',fname)
	f = h.File(diffractionName, 'r')
	d = np.array(f['/data/data'])
	currWavelengthInAngs=f['LCLS']['photon_wavelength_A'][0]
	currDetectorDist=(1.E-3)*f['LCLS']['detectorPosition'][0] 
	f.close()
	angAvgName = ice_dir + fname
	f = h.File(angAvgName, 'r')
	davg = np.array(f['data']['data'][1])
	q = np.array(f['data']['data'][0])
	f.close()
	if mask is not None:
		img_array = np.ma.masked_where(mask==0, d)
	else:
		img_array = d
	# calculate ice peaks
	peak_list = detect_peaks(d, options.thresholdIce, options.peakDimension, mask=mask)
	print ("%s: wavelength:%lf, detectorPos:%lf, peaks:%d"%(re.sub("-angavg.h5",'',fname),currWavelengthInAngs,currDetectorDist, len(peak_list)))
	# calculate sphericity
	sphericity_list = peak_sphericity(d, options.thresholdIce, options.peakDimension)
	# plot peaks
	currImg = img_class(img_array, davg, fname, peakList=peak_list, meanWaveLengthInAngs=currWavelengthInAngs, detectorDistance=currDetectorDist)
	currImg.draw_img_for_tagging()
	if((storeFlag in rangeNumTypes) and not options.inspectOnly):
		waveLengths[storeFlag].append(currWavelengthInAngs)
		avgArr[storeFlag] += d
		avgRadAvg[storeFlag] += davg
		typeOccurences[storeFlag] += 1
		if(not os.path.exists(write_anomaly_dir_types[storeFlag])):
			os.mkdir(write_anomaly_dir_types[storeFlag])

		print ("mv " + angAvgName + " " + write_anomaly_dir_types[storeFlag])
		os.system("mv " + angAvgName + " " + write_anomaly_dir_types[storeFlag])
		os.system("cp " + diffractionName + " " + write_anomaly_dir_types[storeFlag])

#View the averages. Tagging disabled.
for i in range(numTypes):
	if (typeOccurences[i] > 0.):
		print ("averaging new hits in type%s.."%i)
		storeFlag=i
		avgArr[i] /= typeOccurences[i]
		avgRadAvg[i] /= typeOccurences[i]
		typeTag = runtag+'_'+'type'+str(i)
		currImg = img_class(avgArr[i], avgRadAvg[i], typeTag, meanWaveLengthInAngs=np.mean(waveLengths[i]))
		currImg.draw_img_for_viewing()
		(sx,sy) = avgArr[i].shape
		if (not options.inspectOnly):
			f = h.File(write_anomaly_dir_types[i] + typeTag + ".h5", "w")
			entry_1 = f.create_group("/data")
			entry_1.create_dataset("diffraction", data=avgArr[i])
			entry_1.create_dataset("angavg", data=avgRadAvg[i])	
			f.close()
