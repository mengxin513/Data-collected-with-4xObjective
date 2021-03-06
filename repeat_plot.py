
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
    print ('Loading data...')

    microns_per_pixel = 4.25
    df = h5py.File('repeat.hdf5', mode = 'r')
    group = list(df.values())[-1]
    n = len(group)
    pdf = PdfPages('repeatability{}.pdf'.format(group.name.replace('/', '_')))

    dist = np.zeros(n)
    mean_error = np.zeros(n)
    for i in range(n):
        dset = group['distance%03d' % i] #distances
        m = len(dset) - 2
        diff = np.zeros([m, 2])
        move = np.zeros([m, 3])
        for j in range(m):
            data = dset['move%03d' % j] #moves
            init_c = data['init_cam_position']
            final_c = data['final_cam_position']
            init_s = data['init_stage_position']
            moved_s = data['moved_stage_position']
            diff[j, :] = (final_c[:, 1:] - init_c[:, 1:]) * microns_per_pixel
            move[j, :] = moved_s[:] - init_s[:]
        move[:, 0] = move[:, 0] * 0.00960
        move[:, 1] = 0
        move[:, 2] = move[:, 2] * 0.00772
        abs_move = np.sqrt(np.sum(move**2, axis = 1))
        error = np.sqrt(np.sum(diff**2, axis = 1))
        dist[i] = np.mean(abs_move, axis = 0)
        mean_error[i] = np.sqrt(np.mean(error**2))
        print('For plot {}'.format(dist[i]))
        print('Mean position difference is: {}'.format(mean_error[i]))
        print('Minimum position difference is: {}'.format(min(error)))
        print('Maximum position difference is: {}'.format(max(error)))
        matplotlib.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(1, 1)
        ax.plot(diff[:, 0], diff[:, 1], 'ro')
        ax.set_aspect(1)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        r = max(abs(xmin), abs(ymin), abs(xmax), abs(ymax)) * 1.1
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        plt.xlabel(r'X Position [$\mathrm{\mu m}$]', horizontalalignment = 'right', x = 1.0)
        plt.ylabel(r'Y Position [$\mathrm{\mu m}$]', horizontalalignment = 'right', y = 1.0)
        
        pdf.savefig(fig, bbox_inches='tight', dpi=180)
        plt.close('all')

    fig2, ax2 = plt.subplots(1, 1)

    ax2.loglog(dist[4:], mean_error[4:], 'r.')

    ax2.set_xlabel(r'Move Distance [$\mathrm{\mu m}$]')
    ax2.set_ylabel(r'Error [$\mathrm{\mu m}$]')
    pdf.savefig(fig2, bbox_inches='tight', dpi=180)

    pdf.close()
    df.close()
