import numpy as np
import torch


def compute_Inertia_Tensor(coords):
    eye = torch.eye(3, device=coords.device) # [3, 3]
    inner_product = torch.einsum("bi,bi->b", coords, coords) # [N]
    outer_product = torch.einsum("bi,bj->bij", coords, coords)  # [N, 3, 3]

    I_atom = inner_product.unsqueeze(1).unsqueeze(2) * eye.unsqueeze(0) - outer_product  # [N, 3, 3]

    I = torch.mean(I_atom, dim=0)
    return I


def ensure_right_handedness(eigenvectors):
    cross_product = torch.linalg.cross(eigenvectors[:, 0], eigenvectors[:, 1])
    if torch.dot(cross_product, eigenvectors[:, 2]) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]
    return eigenvectors


def determine_handness(vectors):
    # Move tensor to CPU before converting to NumPy
    vectors_cpu = vectors.cpu()
    determinant = np.linalg.det(vectors_cpu)
    if determinant > 0:
        return "Right-handed"
    else:
        return "Left-handed"


def are_points_coplanar(points, tol=1e-6):
    """
    Return whether a set of 3D points is coplanar within a tolerance.
    """
    if points.shape[0] < 4:
        return True

    A, B, C = points[0], points[1], points[2]
    AB = B - A
    AC = C - A
    normal = torch.linalg.cross(AB, AC)

    rest = points[3:]
    ADs = rest - A
    dot_products = torch.matmul(ADs, normal)

    return torch.all(dot_products.abs() < tol)


def find_furthest_point_from_eigenvectors(coord):
    EPS = 1e-6
    mask = (torch.abs(coord[:, 0]) > EPS) & (torch.abs(coord[:, 1]) > EPS)
    
    filtered_coords = coord[mask]
    if filtered_coords.shape[0] == 0:
        return None
        
    distances = torch.norm(filtered_coords, dim=1)
    furthest_index = torch.argmax(distances)
    return filtered_coords[furthest_index]


def find_direction_list(coord):
    # find the further point as the anchor node
    anchor_node = find_furthest_point_from_eigenvectors(coord)
    if anchor_node is None:
        print(f"===== Anchor is None for {coord} =====")
        return None
    if anchor_node[0] > 0:
        if anchor_node[1] > 0:
            direction_list = [1, 1, 1]
        else:  # anchor[1] < 0
            direction_list = [1, -1, -1]
    else: # anchor[1] < 0
        if anchor_node[1] > 0:
            direction_list = [-1, 1, -1]
        else:  # anchor[1] < 0
            direction_list = [-1, -1, 1]
    return direction_list


def build_inertial_frame_and_rotate(cart_coords):
    if are_points_coplanar(cart_coords):
        print(f"========== Coplanar system ==========")
        return None, None
    mass_center = torch.mean(cart_coords, dim=0, keepdim=True)
    cart_coords = cart_coords - mass_center
    
    inertial_tensors = compute_Inertia_Tensor(cart_coords)
    eigen_values, eigen_vectors = torch.linalg.eigh(inertial_tensors)  # [basis 0, basis 1, basis 2]
    rotation = eigen_vectors
    rotation = ensure_right_handedness(rotation)
    handness = determine_handness(rotation)
    assert handness == "Right-handed", "Rotation matrix should be right-hand."

    # NOTE: define the ordering
    EPS = 1e-6
    if abs(eigen_values[0] - eigen_values[1]) < EPS or abs(eigen_values[1] - eigen_values[2]) < EPS:
        print(f"===== tie {eigen_values} =====")

    """
    rotated coords = coords * R
    """
    rotated_cart_coords = torch.matmul(cart_coords, rotation)

    # NOTE: define the directions
    direction_list = find_direction_list(rotated_cart_coords)    
    if direction_list is None:
        return None, None
    rotation[:, 0] *= direction_list[0]
    rotation[:, 1] *= direction_list[1]
    rotation[:, 2] *= direction_list[2]

    rotated_cart_coords = torch.matmul(cart_coords, rotation)

    # NOTE: this is to double-check
    # NOTE: the direction-list here should be [1, 1, 1]
    direction_list = find_direction_list(rotated_cart_coords)
    assert direction_list == [1, 1, 1]

    return rotation, rotated_cart_coords


def cartesian_to_spherical(cart_coords):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        cart_coords: torch.Tensor, shape (N, 3)
        
    Returns:
        torch.Tensor, shape (N, 3)
    """
    x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(y, x)
    phi = torch.acos(z / r)
    
    spherical_coords = torch.stack((r, theta, phi), dim=1)
    return spherical_coords


def spherical_to_cartesian(spherical_coords):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        spherical_coords: torch.Tensor, shape (N, 3)
        
    Returns:
    """
    r, theta_rad, phi_rad = spherical_coords
    x = r * np.sin(phi_rad) * np.cos(theta_rad)
    y = r * np.sin(phi_rad) * np.sin(theta_rad)
    z = r * np.cos(phi_rad)
    return [x, y, z]

# === AF3 Rigid Alignment Implementation (Adapted for PyTorch) ===

def find_optimal_rotation(pred_coords, target_coords):
    """
    Calculates the optimal rotation matrix R to align pred_coords to target_coords.
    Assumes pred_coords and target_coords are already centered.
    Args:
        pred_coords: (N, 3) tensor of centered predicted coordinates.
        target_coords: (N, 3) tensor of centered target coordinates.
    Returns:
        (3, 3) rotation matrix tensor.
    """
    # Calculate cross-covariance matrix H = pred^T @ target
    # Note: AF3 calculates bxt = b.T @ x / N, we omit N as it cancels out in SVD.
    #       Our target is 'b', pred is 'x'. So target.T @ pred.
    cross_cov = torch.matmul(pred_coords.transpose(0, 1), target_coords) # (3, N) @ (N, 3) -> (3, 3)

    # Perform SVD
    try:
        U, S, Vh = torch.linalg.svd(cross_cov)
    except torch._C._LinAlgError as e:
         # Handle potential SVD failure (e.g., if cross_cov is ill-conditioned)
         print(f"Warning: SVD failed: {e}. Returning identity rotation.")
         return torch.eye(3, device=pred_coords.device, dtype=pred_coords.dtype)


    # Calculate rotation matrix R = V @ U.T (Standard Kabsch)
    # AF3 uses R = U @ Vh if SVD is on b.T @ x. Let's verify.
    # If SVD is on H = X^T Y (pred.T @ target), then R = V @ U.T
    # If SVD is on C = Y X^T (target @ pred.T), then R = U @ V^T
    # AF3's bxt = b.T @ x (target.T @ pred), R = U @ V. Let's use R = V @ U.T for robustness.
    V = Vh.transpose(-1, -2)
    R = torch.matmul(V, U.transpose(-1, -2))

    # Ensure right-handed coordinate system (handle reflection)
    det_R = torch.linalg.det(R)
    # Create reflection matrix for correction
    reflection_fix = torch.eye(3, device=R.device, dtype=R.dtype)
    reflection_fix[2, 2] = torch.sign(det_R)

    # Correct rotation matrix: R_corrected = V @ diag(1, 1, sign(det(VU^T))) @ U^T
    R_corrected = torch.matmul(V, torch.matmul(reflection_fix, U.transpose(-1, -2)))

    return R_corrected


def find_optimal_rotation_af3(x, b, allow_reflection=False):
    """
    Find the least squares best fit rotation between two centered sets of N points.
    PyTorch implementation mirroring the logic of the provided NumPy transform_ls.

    Solves Ax = b for A. Where A is the transform rotating x^T into b^T.

    Args:
        x: NxD PyTorch tensor of centered source coordinates (predicted).
        b: NxD PyTorch tensor of centered target coordinates.
        allow_reflection: Whether the returned transformation can reflect.

    Returns:
        DxD rotation matrix tensor A transforming x into b.
    """
    assert x.shape[1] == b.shape[1], f"Dimension mismatch: x({x.shape[1]}) != b({b.shape[1]})"
    assert x.shape[0] == b.shape[0], f"Number of points mismatch: x({x.shape[0]}) != b({b.shape[0]})"
    N, D = x.shape

    # Calculate bxt = b.T @ x / N
    bxt = torch.matmul(b.transpose(0, 1), x) / N

    # Perform SVD on bxt
    try:
        # Ensure input to SVD is float32, as float16 CUDA SVD might not be implemented
        bxt = bxt.float()
        U, S, Vh = torch.linalg.svd(bxt)
    except torch._C._LinAlgError as e:
        print(f"Warning: SVD failed on bxt: {e}. Returning identity rotation.")
        return torch.eye(D, device=x.device, dtype=x.dtype)

    # Calculate initial rotation R = U @ Vh
    R = torch.matmul(U, Vh)

    # Handle reflection if not allowed
    if not allow_reflection:
        # Ensure R is float32 before determinant calculation
        R = R.float()
        det_R = torch.linalg.det(R)
        Vh_corrected = Vh.clone()
        # Multiply the last row of Vh by sign(det(R))
        Vh_corrected[-1, :] *= torch.sign(det_R)
        # Recalculate R with the corrected Vh
        R = torch.matmul(U, Vh_corrected)

    return R


def rigid_align_af3_pytorch_single(x, y):
    """
    Aligns predicted coordinates x to target coordinates y for a single structure.
    PyTorch implementation mirroring the NumPy align function, assuming full, corresponding sequences (L, 3).

    Args:
        x: (L, 3) tensor of predicted coordinates.
        y: (L, 3) tensor of target coordinates.

    Returns:
        (L, 3) tensor of aligned predicted coordinates.
    """
    assert len(x.shape) == 2 and x.shape[1] == 3, f"Expected x shape (L, 3), got {x.shape}"
    assert x.shape == y.shape, "Input shapes must match"
    L = x.shape[0]
    device = x.device
    dtype = x.dtype

    if L < 1: # Handle empty input
        print("Warning: Input coordinates are empty. Returning original coordinates.")
        return x

    # Calculate centroids using ALL points
    x_mean = torch.mean(x, dim=0) # (3,)
    y_mean = torch.mean(y, dim=0) # (3,)

    # Center coordinates
    centered_x = x - x_mean # (L, 3)
    centered_y = y - y_mean # (L, 3)

    # Find optimal rotation
    # Use the AF3-style function, pass the centered coordinates
    R = find_optimal_rotation_af3(centered_x, centered_y, allow_reflection=False)

    # Apply rotation to the centered predicted coordinates
    transformed_centered_x = torch.matmul(centered_x, R.transpose(0, 1)) # (L, 3) @ (3, 3) -> (L, 3)

    # Add target centroid to translate
    aligned_x = transformed_centered_x + y_mean # (L, 3)

    return aligned_x
