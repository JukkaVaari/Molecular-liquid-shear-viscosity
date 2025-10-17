import sys

from aiida import orm
from aiida.cmdline.params import arguments, types
import click
import numpy as np

from . import cmd_data


@cmd_data.command('plot-viscosity')
@arguments.DATUM(
    'data',
    required=True,
    type=types.DataParamType(sub_classes=('aiida.data:core.array',))
)
@click.option(
    '-s', '--show-plot',
    is_flag=True,
    default=False,
    help='Show the plot interactively.'
)
@click.option(
    '-o', '--output-file',
    type=click.Path(dir_okay=False, writable=True),
    default='viscosity_fit.png',
    show_default=True,
    help='Path to save the output plot image.'
)
def plot_viscosity(
        data: orm.ArrayData,
        output_file: str,
        show_plot: bool = False,
    ):
    """Plot viscosity data and fit to the Eyring model."""
    from aitw_aiida_viscosity.viscosity import eyring_viscosity, fit_viscosity_eyring

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as e:
        click.echo('Please install the package with [plotting] extras to use this command.')
        sys.exit(1)

    shear_rates = data.get_array('shear_rates')
    viscosities = data.get_array('viscosities')

    sucess, eta_N, sigma_E = fit_viscosity_eyring(shear_rates, viscosities)
    if not sucess:
        click.echo('Curve fitting failed.')
        sys.exit(1)
    click.echo(f"Fit successful: ")
    for var, value in [('eta_N', eta_N), ('sigma_E', sigma_E)]:
        click.echo(f'{var:>20s} = {value:13.6e}')

    if show_plot:
        messages = []
        for backend in ['Qt5Agg', 'TkAgg']:
            try:
                matplotlib.use(backend)
            except Exception as e:
                messages.append(f"Could not use matplotlib backend '{backend}': {e}")
                continue
            else:
                break
        else:
            for msg in messages:
                click.echo(msg)
            click.echo('Could not set a suitable matplotlib backend to show the plot.')
            show_plot = False

    # Create plot
    fig, ax = plt.subplots()
    ax.loglog(shear_rates, viscosities, 'o', label='MD Data')

    shear_rates_fit = np.logspace(np.log10(min(shear_rates)), np.log10(max(shear_rates)), 200)
    ax.loglog(
        shear_rates_fit,
        eyring_viscosity(shear_rates_fit, eta_N, sigma_E),
        '-', label=f'Fit: $\\eta_N$={eta_N:.2f} mPa·s, $\\sigma_E$={sigma_E:.2e} s'
    )

    ax.set_xlabel('Shear Rate (1/s)')
    ax.set_ylabel('Viscosity (mPa·s)')
    ax.set_title('Viscosity vs Shear Rate (log-log)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--')

    plt.tight_layout()
    fig.savefig(output_file)
    click.echo(f"Plot saved locally as '{output_file}'.")

    if show_plot:
        plt.show()
