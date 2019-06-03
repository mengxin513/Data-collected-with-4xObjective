
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import math

if __name__ == "__main__":
    print ('Loading data...')

    # Loads data from the HDF5 file
    df = h5py.File('precision.hdf5', mode = 'r')
    with PdfPages('precision_all.pdf') as pdf:
        for group in df.values():
            dset = list(group.values())[-1]
            n = len(dset)

            average = np.zeros([math.trunc(n / 10), 3])

            j = 0
            for i in range(n):
                if (i + 1) % 10 == 0:
                    average[j, :] = np.mean(dset[(i + 1 - 10):i, :], axis = 0)
                    j = j + 1

            matplotlib.rcParams.update({'font.size': 12})

            microns_per_pixel = 4.25

            t1 = dset[:, 0]
            x1 = dset[:, 1] * microns_per_pixel
            x1 -= np.mean(x1)
            y1 = dset[:, 2] * microns_per_pixel
            y1 -= np.mean(y1)

            t2 = average[:, 0]
            x2 = average[:, 1] * microns_per_pixel
            x2 -= np.mean(x2)
            y2 = average[:, 2] * microns_per_pixel
            y2 -= np.mean(y2)

            fig1, ax1 = plt.subplots(1, 1)

            ax1.plot(t1, x1, 'r-', alpha = 0.5)
            line1, = ax1.plot(t2, x2, 'r-')
            ax2 = ax1.twinx()
            ax2.plot(t1, y1, 'b-', alpha = 0.5)
            line2, = ax2.plot(t2, y2, 'b-')
            
            # Make the scale the same for X and Y (so it's obvious which is moving)
            xmin, xmax = ax1.get_ylim()
            ymin, ymax = ax2.get_ylim()
            r = max(xmax - xmin, ymax - ymin)
            ax1.set_ylim((xmax + xmin)/2 - r/2, (xmax + xmin)/2 + r/2)
            ax2.set_ylim((ymax + ymin)/2 - r/2, (ymax + ymin)/2 + r/2)

            ax1.set_xlabel(r'Time [$\mathrm{s}$]')
            ax1.set_ylabel(r'X Position [$\mathrm{\mu m}$]')
            ax2.set_ylabel(r'Y Position [$\mathrm{\mu m}$]')

            ax1.legend((line1, line2), ('X', 'Y'), loc = 'lower right')

            plt.tight_layout()

            pdf.savefig(fig1)
            plt.close(fig1)

            fig2, ax3 = plt.subplots(1, 1)

            ax3.plot(x1, y1, '.')
            ax3.set_aspect(1)

            #plot the XY motion, make the limits equal
            xmin, xmax = ax3.get_xlim()
            ymin, ymax = ax3.get_ylim()
            r = max(xmax - xmin, ymax - ymin)
            ax3.set_xlim((xmax + xmin)/2 - r/2, (xmax + xmin)/2 + r/2)
            ax3.set_ylim((ymax + ymin)/2 - r/2, (ymax + ymin)/2 + r/2)

            ax3.set_xlabel(r'X Position [$\mathrm{\mu m}$]')
            ax3.set_ylabel(r'Y Position [$\mathrm{\mu m}$]')

            plt.tight_layout()

            pdf.savefig(fig2)
            plt.close(fig2)
    df.close()
