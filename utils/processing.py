import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter


def zero_mean_unit_var(image, mask=None, fill_value=0):
    """Normalizes an image to zero mean and unit variance."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    mean = np.mean(img_array[msk_array>0])
    std = np.std(img_array[msk_array>0])

    if std > 0:
        img_array = (img_array - mean) / std
        img_array[msk_array==0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def range_matching(image, mask=None, low_percentile=4, high_percentile=96, fill_value=0):
    """Normalizes an image by mapping the low_percentile to zero, and the high_percentile to one."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    lo_p = np.percentile(img_array[msk_array>0], low_percentile)
    hi_p = np.percentile(img_array[msk_array>0], high_percentile)

    img_array = (img_array - lo_p) / (hi_p - lo_p)
    img_array[msk_array == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def zero_one(image, mask=None, fill_value=0):
    """Normalizes an image by mapping the min to zero, and max to one."""
    # sitk.WriteImage(image, 'test.nii.gz')
    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)
    img_array_keep = np.array(list(img_array))
    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    # value_min = np.min(img_array[msk_array>0])
    # value_max = np.max(img_array[msk_array>0])
    data_one = img_array.reshape(-1)
    data_one.sort()

    img_array_keep == img_array

    max_value = data_one[int(len(data_one) * 0.99)]
    min_value = data_one[int(len(data_one) * 0.01)]

    img_array_keep[(img_array_keep > max_value)] = max_value
    img_array_keep[(img_array_keep < min_value)] = min_value
    img_array_keep = (img_array_keep - min_value) / (max_value - min_value)
    img_array_keep[img_array_keep == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array_keep)
    image_normalised.CopyInformation(image)
    # sitk.WriteImage(image_normalised, 'test_norm.nii.gz')
    return image_normalised


def threshold_zero(image, mask=None, fill_value=0):
    """Thresholds an image at zero."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array > 0
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    img_array[msk_array == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def same_image_domain(image1, image2):
    """Checks whether two images cover the same physical domain."""

    same_size = image1.GetSize() == image2.GetSize()
    same_spacing = image1.GetSpacing() == image2.GetSpacing()
    same_origin = image1.GetOrigin() == image2.GetOrigin()
    same_direction = image1.GetDirection() == image2.GetDirection()

    return same_size and same_spacing and same_origin and same_direction


def reorient_image(image):
    """Reorients an image to standard radiology view."""

    dir = np.array(image.GetDirection()).reshape(len(image.GetSize()), -1)
    ind = np.argmax(np.abs(dir), axis=0)
    new_size = np.array(image.GetSize())[ind]
    new_spacing = np.array(image.GetSpacing())[ind]
    new_extent = new_size * new_spacing
    new_dir = dir[:, ind]

    flip = np.diag(new_dir) < 0
    flip_diag = flip * -1
    flip_diag[flip_diag == 0] = 1
    flip_mat = np.diag(flip_diag)

    new_origin = np.array(image.GetOrigin()) + np.matmul(new_dir, (new_extent * flip))
    new_dir = np.matmul(new_dir, flip_mat)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing.tolist())
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(new_dir.flatten().tolist())
    resample.SetOutputOrigin(new_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    return resample.Execute(image)


def resample_image_to_ref(image, ref, is_label=False, pad_value=0):
    """Resamples an image to match the resolution and size of a given reference image."""

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image)


def resample_image(image, add_spacing=None, out_spacing=(1.0, 1.0, 1.0), out_size=None, is_label=False, pad_value=0):
    """Resamples an image to given element spacing and output size."""

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    if add_spacing is None:
        out_spacing = list(np.array(out_spacing))
    else:
        out_spacing = list(np.array(out_spacing)*add_spacing)
    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)
    out_origin = 0
    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing     # 原始原点在world space的坐标- image.GetOrigin()
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)      # 保持direction不变

    out_origin = (np.array(image.GetOrigin()) + original_center) - out_center       # the first term is the coordinate of original center at world space.
    # out_origin_1 = np.array(image.GetOrigin()) + (out_center - original_center)  # 原点没有spacing的问题

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)


    # resample1 = sitk.ResampleImageFilter()
    # resample1.SetOutputSpacing(out_spacing)
    # resample1.SetSize(out_size.tolist())
    # resample1.SetOutputDirection(image.GetDirection())
    # resample1.SetOutputOrigin(out_origin_1.tolist())
    # resample1.SetTransform(sitk.Transform())
    # resample1.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        # resample1.SetInterpolator(sitk.sitkNearestNeighbor)
        return resample.Execute(image)      #, resample1.Execute(image)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
        if image.GetNumberOfComponentsPerPixel() == 1:
            return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))
        else:
            return resample.Execute(sitk.Cast(image, sitk.sitkVectorFloat32))


def extract_patch(image, pixel, out_spacing=(1.0, 1.0, 1.0), out_size=(32, 32, 32), is_label=False, pad_value=0):
    """Extracts a patch of given resolution and size at a specific location."""

    original_spacing = np.array(image.GetSpacing())

    out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    pos = np.matmul(original_direction, np.array(pixel) * np.array(original_spacing)) + np.array(image.GetOrigin())
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(pos - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    if image.GetNumberOfComponentsPerPixel() == 1:
        return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))
    else:
        return resample.Execute(sitk.Cast(image, sitk.sitkVectorFloat32))


def one_hot_labelmap(labelmap, sub_labelmap=None, smoothing_sigma=0):
    """Converts a single channel labelmap to a one-hot labelmap."""
    lab_array = sitk.GetArrayFromImage(labelmap)
    if sub_labelmap is None:
        labels = np.unique(lab_array)
        labels.sort()
    else:
        labels_all = np.unique(lab_array)
        labels_all.sort()
        labels = sub_labelmap

        remove_labels = [i for i in range(len(labels_all)) if labels_all[i] not in labels]
        """remove labels all move to label 0"""

    labelmap_size = list(labelmap.GetSize()[::-1])
    labelmap_size.append(labels.size)
    dist = np.zeros(labels.size)

    lab_array_one_hot = np.zeros(labelmap_size).astype(float)
    for idx, lab in enumerate(labels):
        if smoothing_sigma > 0:
            lab_array_one_hot[..., idx] = gaussian_filter((lab_array == lab).astype(float), sigma=smoothing_sigma, mode='nearest')
        else:
            if idx ==0:
                assert sub_labelmap[0] == 0
                bg_no = (lab_array == lab).astype(float).sum()
                bg_map = lab_array == lab
                for j in remove_labels:
                    bg_no += (lab_array == j).astype(float).sum()
                    bg_map = np.logical_or(bg_map, lab_array==j)
                dist[0] = bg_no
                lab_array_one_hot[..., 0] = bg_map
            else:
                dist[idx] = (lab_array == lab).astype(float).sum()
                lab_array_one_hot[..., idx] = lab_array == lab

    labelmap_one_hot = sitk.GetImageFromArray(lab_array_one_hot, isVector=True)
    labelmap_one_hot.CopyInformation(labelmap)
    # print(labelmap_one_hot.GetSize())
    return labelmap_one_hot, dist