from observables_builder import *


# for test
vis_a_p4_cm = vector.Vector(x=5, y=5, z=0, t=8)
tau_a_p4_cm = vector.Vector(x=7, y=1, z=7, t=13)
helicity_basis_a = helicity_basis(tau_a_p4_cm)
print("Original helicity basis:")
for axis in ['k', 'r', 'n']:
    print(f"{axis}: {helicity_basis_a[axis]}")
boost_to_a_rest = -tau_a_p4_cm.to_beta3()

boosted_helicity_basis_a = {}
print("Boost helicity basis to tau a rest frame for test:")
for axis in ['k', 'r', 'n']:
    axis_vector = helicity_basis_a[axis]
    helicity_basis_a_tmp = vector.Vector(
        x=axis_vector.x,
        y=axis_vector.y,
        z=axis_vector.z,
        t=0
    )
    print(f"Before boost {axis}: {helicity_basis_a_tmp}")
    helicity_basis_a_a_rest = helicity_basis_a_tmp.boost(boost_to_a_rest)
    print(f"After boost {axis}: {helicity_basis_a_a_rest}")
    boosted_helicity_basis_a[axis] = helicity_basis_a_a_rest.to_pxpypz().unit()
    print(f"{axis}: {boosted_helicity_basis_a[axis]}")
    print()

print("==================================\n\n")

vis_a_p4_a_rest = vis_a_p4_cm.boost(boost_to_a_rest)
print(f"Before boost visible momentum: {vis_a_p4_cm}, {vis_a_p4_cm.to_pxpypz().unit()}")
print(f"After boost visible momentum: {vis_a_p4_a_rest}, {vis_a_p4_a_rest.to_pxpypz().unit()}")
print("Visible momentum in tau a rest frame:", vis_a_p4_a_rest)
print("Not boost helicity basis to tau a rest frame for test:")
for axis in ['k', 'r', 'n']:
    print(vis_a_p4_a_rest.to_pxpypz().unit().dot(helicity_basis_a[axis]))

print("Boost helicity basis to tau a rest frame for test:")
for axis in ['k', 'r', 'n']:
    print(vis_a_p4_a_rest.to_pxpypz().unit().dot(boosted_helicity_basis_a[axis]))