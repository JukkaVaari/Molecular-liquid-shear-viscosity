"""Collection of calcfunctions used by the workchains."""
import re

from aiida import orm
from aiida.engine import calcfunction
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ..viscosity import fit_viscosity_eyring


def string_safe_float(value: float) -> str:
    """Convert a float to an aiida-context safe string"""
    return str(value).replace('.', '_').replace('-', 'm')

@calcfunction
def extract_files_suffix(folder: orm.FolderData, suffix: orm.Str) -> orm.SinglefileData:
    """Extract a file with a specific suffix from a FolderData node."""
    suffix = suffix.value
    for file_obj in folder.list_objects():
        filename = file_obj.name
        if filename.endswith(suffix):
            with folder.open(filename, 'rb') as f:
                out_node = orm.SinglefileData(file=f, filename=filename)
            return out_node
    raise ValueError(f"No file with suffix '{suffix}' found in the provided FolderData.")

@calcfunction
def run_resp_injection(
        pdb_file: orm.SinglefileData, itp_file: orm.SinglefileData
    ) -> orm.SinglefileData:
    """
    Inject RESP charges from a PDB file into an ITP file.

    Args:
        pdb_file (SinglefileData): The PDB file containing RESP charges.
        itp_file (SinglefileData): The ITP file to be updated with RESP charges.

    Returns:
        SinglefileData: The updated ITP file with RESP charges injected.
    """
    # Retrieve file contents

    with pdb_file.open() as f_pdb:
        pdb_lines = f_pdb.readlines()

    with itp_file.open() as f_itp:
        itp_lines = f_itp.readlines()

    # Extract RESP charges from the PDB file (columns 71-76)
    charges = []
    for line in pdb_lines:
        if line.startswith(('ATOM', 'HETATM')):
            try:
                charge = float(line[70:76].strip())
                charges.append(charge)
            except ValueError:
                continue  # Skip any malformed lines

    # Inject charges into ITP file
    updated_lines = []
    in_atoms_section = False
    charge_index = 0
    for line in itp_lines:
        if line.strip().startswith('[ atoms ]'):
            in_atoms_section = True
            updated_lines.append(line)
            continue
        if in_atoms_section:
            if line.strip().startswith('['):  # End of atoms section
                in_atoms_section = False
            elif line.strip() and not line.strip().startswith(';'):
                fields = re.split(r'\s+', line.strip())
                if len(fields) >= 7 and charge_index < len(charges):
                    fields[6] = f"{charges[charge_index]:.6f}"
                    charge_index += 1
                    updated_lines.append('    '.join(fields) + '\n')
                    continue
        updated_lines.append(line)

    updated_itp_node = orm.SinglefileData.from_string('\n'.join(updated_lines), filename='updated.itp')

    return updated_itp_node

@calcfunction
def update_top_file(
        nmols: orm.Int,
        top_file: orm.SinglefileData,
        itp_file: orm.SinglefileData,
    ) -> orm.SinglefileData:
    """
    Update the .top file to reference the new .itp file and adjust the molecule count.

    Args:
        nmols (Int): Number of molecules.
        top_file (SinglefileData): The original .top file to be updated.
        itp_file (SinglefileData): The updated .itp file with RESP charges.
    Returns:
        SinglefileData: The updated .top file.
    """
    nmols = nmols.value
    with top_file.open() as f:
        lines = f.readlines()

    # Determine the updated .itp filename
    updated_itp_filename = itp_file.filename

    # Process lines: update .itp reference and [ molecules ] count
    new_lines = []
    in_molecules_section = False
    for line in lines:
        stripped = line.strip()

        # Replace .itp file reference
        if stripped.startswith('#include') and stripped.endswith('.itp"'):
            # Replace with updated itp filename
            newline = f'#include "{updated_itp_filename}"\n'
            new_lines.append(newline)
            continue

        # Update molecule count
        if '[ molecules ]' in stripped:
            in_molecules_section = True
            new_lines.append(line)
            continue
        elif in_molecules_section:
            if stripped and not stripped.startswith(';'):
                parts = stripped.split()
                if len(parts) >= 2:
                    parts[1] = str(nmols)  # Update molecule count
                    newline = f'{parts[0]:<20s}{parts[1]}\n'
                    new_lines.append(newline)
                    in_molecules_section = False  # Done with this section
                    continue

        new_lines.append(line)

    top_updated = orm.SinglefileData.from_string('\n'.join(new_lines), filename='updated.top')

    return top_updated

@calcfunction
def get_box_size(
        nmols: orm.Int,
        smiles_string: orm.Str,
    ):
    """Approximate molar mass from SMILES string using RDKit and store as AiiDA Float node."""
    smiles = smiles_string.value
    mol = Chem.MolFromSmiles(smiles)
    nmols = nmols.value

    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    mw = rdMolDescriptors.CalcExactMolWt(mol)

    Na = 6.022e23
    rho = 0.5 # g/cm3 - small enough so that gmx insert-molecule works with ease
    box_volume_cm3 = nmols * mw / (rho * Na)
    box_volume_nm3 = box_volume_cm3 * 1e21
    edge_length_nm = box_volume_nm3 ** (1/3)

    return  orm.Float(edge_length_nm)

@calcfunction
def generate_veloxchem_input(basis_set: orm.Str) -> orm.SinglefileData:
    """Generate a basic VeloxChem input file."""
    template = '\n'.join([
        'import sys',
        'import veloxchem as vlx',
        'infile = sys.argv[1]',
        'molecule = vlx.Molecule.read_xyz(infile)',
        'mol_xyz = molecule.get_xyz_string()',
        # 'basis = vlx.MolecularBasis.read(molecule, "6-31G*")',
        f'basis = vlx.MolecularBasis.read(molecule, "{basis_set.value}")',
        'resp_drv = vlx.RespChargesDriver()',
        'resp_charges = resp_drv.compute(molecule, basis, "resp")',
    ])

    return orm.SinglefileData.from_string(template, filename='aiida_vlx.py')


@calcfunction
def generate_gromacs_minimization_input(minimization_steps: orm.Int) -> orm.SinglefileData:
    """Generate a basic GROMACS minimization input file."""
    template = '\n'.join([
        'integrator          = steep',
        f'nsteps              = {minimization_steps.value}',
        'nstcgsteep          = 100',
        'emtol               = 0',
        'emstep              = 0.01',
        'nstlog              = 100',
        'nstenergy           = 100',
        'nstlist             = 10',
        'ns_type             = grid',
        'pbc                 = xyz',
        'cutoff-scheme       = Verlet',
        'vdwtype             = cutoff',
        'vdw-modifier        = None',
        'rlist               = 1.0',
        'rvdw                = 1.0',
        'rvdw-switch         = 1.0',
        'coulombtype         = PME',
        'rcoulomb            = 1.0',
        'DispCorr            = EnerPres'
    ])

    return orm.SinglefileData.from_string(template, filename='minim.mdp')

@calcfunction
def generate_gromacs_equilibration_input(
        num_steps: orm.Int,
        time_step: orm.Float,
        reference_temperature: orm.Float,
    ):
    """Generate a basic GROMACS equilibration input file."""

    template = '\n'.join([
        'integrator          = md-vv',
        f'nsteps              = {num_steps.value}',
        f'dt                  = {time_step.value}',
        'comm_mode           = linear',
        'nstcomm             = 100',
        'nstxout-compressed  = 100000',
        'nstvout             = 0',
        'nstlog              = 1000',
        'nstenergy           = 10000',
        'nstlist             = 50',
        'ns_type             = grid',
        'pbc                 = xyz',

        'constraints         = none',
        'cutoff-scheme       = Verlet',
        'vdwtype             = Cut-off',
        'vdw-modifier        = None',
        'rlist               = 1.0',
        'rvdw                = 1.0',
        'rvdw-switch         = 1.0',
        'coulombtype         = PME',
        ';coulombtype         = Cut-off',
        'rcoulomb            = 1.0',
        'DispCorr            = EnerPres',

        f'ref_t               = {reference_temperature.value}',
        'Tcoupl              = v-rescale',
        'tc-grps	            = system',
        'tau_t               = 1.0',
        'Pcoupl              = C-rescale',
        ';Pcoupl              = no',
        'Pcoupltype          = isotropic',
        'tau_p               = 5.0',
        'compressibility     = 4.5e-5',
        'ref_p               = 1.0 ',
        'refcoord_scaling    = all',
    ])

    return orm.SinglefileData.from_string(template, filename='equilibrate.mdp')

def _generate_gromacs_deform_vel_input(nsteps: int, time_step: float, ref_t: float, deform_vel: float):
    """Generate a basic GROMACS shear rate input file."""
    template = '\n'.join([
        'integrator          = md',
        f'nsteps              = {nsteps}',
        f'dt                  = {time_step}',
        'nstxout-compressed  = 3000000',
        'nstvout             = 0',
        'nstlog              = 1000',
        'nstenergy           = 50',
        'nstcalcenergy       = 5',
        'nstlist             = 50',
        'ns_type             = grid',
        'pbc                 = xyz',

        'constraints         = none',
        'cutoff-scheme       = Verlet',
        'vdwtype             = Cut-off',
        'vdw-modifier        = None',
        'rlist               = 1.0',
        'rvdw                = 1.0',
        'rvdw-switch         = 1.0',
        'coulombtype         = PME',
        ';coulombtype         = Cut-off',
        'rcoulomb            = 1.0',
        'DispCorr            = EnerPres',

        f'ref_t               = {ref_t}',
        'Tcoupl              = v-rescale',
        'tc-grps             = system',
        'tau_t               = 1.0',
        'Pcoupl              = no',

        f'deform              = 0.0 0.0 0.0 {deform_vel} 0.0 0.0',
        'deform-init-flow    = yes',
    ])

    return template

@calcfunction
def generate_gromacs_deform_vel_inputs(
        nsteps: orm.Int,
        time_step: orm.Float,
        ref_t: orm.Float,
        deform_velocities: orm.List,
    ) -> dict[str, orm.SinglefileData]:
    """Generate a basic GROMACS shear rate input file."""
    res = {}
    for defvel in deform_velocities:
        str_defvel = string_safe_float(defvel)
        template = _generate_gromacs_deform_vel_input(
            nsteps=nsteps.value,
            time_step=time_step.value,
            ref_t=ref_t.value,
            deform_vel=defvel,
        )
        res[f'mdp_{str_defvel}'] = orm.SinglefileData.from_string(template, filename='aiida.mdp')

    return res

@calcfunction
def extract_deformation_velocities(mdp_files):
    """Extract deformation velocities from MDP files."""
    velocities = []
    for file_path in mdp_files.get_list():
        with open(file_path, 'r') as f:
            lines = f.readlines()
            content = ''.join(lines)
            match = re.search(r'deform\s*=\s*([-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+)', content)
            if not match:
                raise ValueError(f"No 'deform' line found in {file_path}!")
            numbers = [float(x) for x in match.group(1).split()]
            deform_velocity = numbers[3]  # 4th number (z direction)
            velocities.append(deform_velocity)  # fourth line
    return orm.List(list=velocities)

@calcfunction
def extract_box_length(grofile: orm.SinglefileData) -> orm.Float:
    """Extract the box length from a .gro file."""
    content = grofile.get_content()
    _, last_line = content.rstrip().rsplit('\n', 1)
    box_length_x = float(list(filter(None, last_line.split()))[0])
    return orm.Float(box_length_x)

@calcfunction
def extract_pressure_from_xvg(xvg_file: orm.SinglefileData) -> orm.List:
    """Extract pressure values from a GROMACS .xvg file."""
    with xvg_file.open() as file_handle:
        data = np.loadtxt(file_handle, comments=['@', '#'])
    avg_pressure = - np.mean(data[:, 1])  # Convert to a positive value
    return orm.Float(avg_pressure)

@calcfunction
def join_pressure_results(
        deform_vel: orm.List,
        **pressure_results,
    ) -> orm.List:
    """Join pressure results from multiple calculations into a single list."""
    pressures = []
    for defvel in deform_vel:
        str_defvel = string_safe_float(defvel)
        pressure = pressure_results.get(f'pressure_{str_defvel}', None)
        if pressure is None:
            raise ValueError(f"Missing pressure result for deformation velocity {defvel}!")
        pressures.append(pressure.value)

    return orm.List(list=pressures)

@calcfunction
def compute_viscosities(
        deformation_velocities: orm.List,
        pressures: orm.List,
        box_length: orm.Float,
    ) -> orm.ArrayData:
    """Compute shear rates from deformation velocities and box length."""
    box_length_nm = box_length.value

    deform_vel = []
    shear_rates = []
    viscosities = []

    for deform_vel_nm_per_ps, pressure_bar in zip(deformation_velocities.get_list(), pressures.get_list()):
        shear_rate = (deform_vel_nm_per_ps * 1000) / (box_length_nm * 1e-9)  # [1/s]
        pressure_Pa = pressure_bar * 1e5  # [Pa]
        viscosity_Pa_s = pressure_Pa / shear_rate  # [Pa.s]
        viscosity_mPa_s = viscosity_Pa_s * 1000  # [mPa.s]

        deform_vel.append(deform_vel_nm_per_ps)
        shear_rates.append(shear_rate)
        viscosities.append(viscosity_mPa_s)

    # Ensure arrays are sorted by increasing deformation velocity
    order_args = np.argsort(deform_vel)

    def_vels = np.array(deform_vel)[order_args]
    srate_array = np.array(shear_rates)[order_args]
    visc_array = np.array(viscosities)[order_args]
    pressures_array = np.array(pressures.get_list())[order_args]

    array = orm.ArrayData()
    array.set_array('deformation_velocities', def_vels)
    array.set_array('pressure_averages', pressures_array)
    array.set_array('shear_rates', srate_array)
    array.set_array('viscosities', visc_array)

    return array

@calcfunction
def fit_viscosity(
        viscosity_data: orm.ArrayData,
    ) -> dict[str, orm.Float]:
    """Fit viscosity data to the Eyring model."""
    shear_rates = viscosity_data.get_array('shear_rates')
    viscosities = viscosity_data.get_array('viscosities')

    success, eta_N, sigma_E = fit_viscosity_eyring(shear_rates, viscosities)

    if not success:
        raise ValueError('Curve fitting to the Eyring model failed!')

    res = {
        'eta_N': orm.Float(eta_N),
        'sigma_E': orm.Float(sigma_E),
    }

    return res
