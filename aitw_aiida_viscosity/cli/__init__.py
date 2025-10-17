# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
"""Module for the command line interface."""
from aiida.cmdline.groups import VerdiCommandGroup
from aiida.cmdline.params import options, types
import click


@click.group(
    'aitw-aiida-viscosity',
    cls=VerdiCommandGroup,
    context_settings={'help_option_names': ['-h', '--help']}
)
@options.PROFILE(type=types.ProfileParamType(load_profile=True), expose_value=False)
def cmd_root():
    """CLI for the `AITW-aiida-viscosity` plugin."""

# Apply the ``tui`` interface if the dependency is installed. If the case, this will add the ``verdi tui`` command
# that allows to explore the interface in a graphical manner in the terminal.
try:
    from trogon import tui
except ImportError:
    pass
else:
    tui()(cmd_root)


from .data import plot_viscosity
from .workflows import cmd_workflow

__all__ = (
    'cmd_root',
    'cmd_workflow',
    'plot_viscosity',
)
