import typer

class Kepler:
    def __init__(self):
        pass

    def scspatial(adata, *, basis: str="spatial", img: Union[np.ndarray, None] = None, img_key: Union[str, None, Empty] = _empty,
            library_id: Union[str, Empty] = _empty, crop_coord: Tuple[int, int, int, int] = None,alpha_img: float = 1.0, bw: Optional[bool] = False,
            size: float = 1.0,scale_factor: Optional[float] = None,spot_size: Optional[float] = None,na_color: Optional[ColorLike] = None,
            show: Optional[bool] = None,return_fig: Optional[bool] = None,save: Union[bool, str, None] = None,gdf: Optional[object] = None,gdf_colors: Optional[dict] = None,gdf_alpha: float = 1.0,**kwargs,) -> Union[Axes, List[Axes], None]:
        """
        Scatter plot in spatial coordinates.
        This function allows overlaying data on top of images.
        Use the parameter `img_key` to see the image in the background
        And the parameter `library_id` to select the image.
        By default, `'hires'` and `'lowres'` are attempted.

        Use `crop_coord`, `alpha_img`, and `bw` to control how it is displayed.
        Use `size` to scale the size of the Visium spots plotted on top.

        As this function is designed to for imaging data, there are two key assumptions
        about how coordinates are handled:

        1. The origin (e.g `(0, 0)`) is at the top left – as is common convention
        with image data.

        2. Coordinates are in the pixel space of the source image, so an equal
        aspect ratio is assumed.

        If your anndata object has a `"spatial"` entry in `.uns`, the `img_key`
        and `library_id` parameters to find values for `img`, `scale_factor`,
        and `spot_size` arguments. Alternatively, these values be passed directly.

        Parameters
        ----------
        {adata_color_etc}
        {scatter_spatial}
        {scatter_bulk}
        {show_save_ax}

        Returns
        -------
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

        Examples
        --------
        This function behaves very similarly to other embedding plots like
        :func:`~scanpy.pl.umap`

        """
        # get default image params if available
        library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
        img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
        spot_size = _check_spot_size(spatial_data, spot_size)
        scale_factor = _check_scale_factor(
            spatial_data, img_key=img_key, scale_factor=scale_factor
        )
        crop_coord = _check_crop_coord(crop_coord, scale_factor)
        na_color = _check_na_color(na_color, img=img)

        if bw:
            cmap_img = "gray"
        else:
            cmap_img = None
        circle_radius = size * scale_factor * spot_size * 0.5

        axs = embedding(
            adata,
            basis=basis,
            scale_factor=scale_factor,
            size=circle_radius,
            na_color=na_color,
            show=False,
            save=False,
            **kwargs,
                )
        if not isinstance(axs, list):
            axs = [axs]

        if gdf_colors:
            gdf_classes=pd.DataFrame(gdf['classification'].tolist())
            for anno_class, anno_color in gdf_colors.items():
                class_inds = gdf_classes[gdf_classes['name'] == anno_class].index
                gdf_c = gdf.iloc[class_inds,:]
                gdf_c.plot(color=anno_color, ax=axs[0], aspect=1, alpha=gdf_alpha)
        elif gdf:
            gdf.plot(color='red', ax=axs[0], aspect=1, alpha=gdf_alpha)
        for ax in axs:
            cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
            if img is not None:
                ax.imshow(img, cmap=cmap_img, alpha=alpha_img)
            else:
                ax.set_aspect("equal")
                ax.invert_yaxis()
            if crop_coord is not None:
                ax.set_xlim(crop_coord[0], crop_coord[1])
                ax.set_ylim(crop_coord[3], crop_coord[2])
            else:
                ax.set_xlim(cur_coords[0], cur_coords[1])
                ax.set_ylim(cur_coords[3], cur_coords[2])
        _utils.savefig_or_show('show', show=show, save=save)
        if show is False or return_fig is True:
            return axs

    def _check_scale_factor(
        spatial_data: Optional[Mapping],
        img_key: Optional[str],
        scale_factor: Optional[float],
    ) -> float:
        """Resolve scale_factor, defaults to 1."""
        if scale_factor is not None:
            return scale_factor
        elif spatial_data is not None and img_key is not None:
            return spatial_data['scalefactors'][f"tissue_{img_key}_scalef"]
        else:
            return 1.0

    def _check_crop_coord(
        crop_coord: Optional[tuple],
        scale_factor: float,
    ) -> Tuple[float, float, float, float]:
        """Handle cropping with image or basis."""
        if crop_coord is None:
            return None
        if len(crop_coord) != 4:
            raise ValueError("Invalid crop_coord of length {len(crop_coord)}(!=4)")
        crop_coord = tuple(c * scale_factor for c in crop_coord)
        return crop_coord

    def _check_na_color(
        na_color: Optional[ColorLike], *, img: Optional[np.ndarray] = None
    ) -> ColorLike:
        if na_color is None:
            if img is not None:
                na_color = (0.0, 0.0, 0.0, 0.0)
            else:
                na_color = "lightgray"
        return na_color

    def _check_img(
        spatial_data: Optional[Mapping],
        img: Optional[np.ndarray],
        img_key: Union[None, str, Empty],
        bw: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Resolve image for spatial plots.
        """
        if img is None and spatial_data is not None and img_key is _empty:
            img_key = next(
                    (k for k in ['hires', 'lowres'] if k in spatial_data['images']),
            )
        if img is None and spatial_data is not None and img_key is not None:
            img = spatial_data["images"][img_key]
        if bw:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        return img, img_key

    def _check_spot_size(
        spatial_data: Optional[Mapping], spot_size: Optional[float]
        ) -> float:
        """
        Resolve spot_size value.
        This is a required argument for spatial plots.
        """
        if spatial_data is None and spot_size is None:
            raise ValueError(
                    "When .uns['spatial'][library_id] does not exist, spot_size must be "
                    "provided directly."
                    )
        elif spot_size is None:
            return spatial_data['scalefactors']['spot_diameter_fullres']
        else:
            return spot_size

    def _check_spatial_data(uns: Mapping, library_id: Union[Empty, None, str]) -> Tuple[Optional[str], Optional[Mapping]]:
        """
        Given a mapping, try and extract a library id/ mapping with spatial data.
        Assumes this is `.uns` from how we parse visium data.
        """
        spatial_mapping = uns.get("spatial", {})
        if library_id is _empty:
            if len(spatial_mapping) > 1:
                raise ValueError("Found multiple possible libraries in `.uns['spatial']. Please specify."
                                f" Options are:\n\t{list(spatial_mapping.keys())}")
            elif len(spatial_mapping) == 1:
                library_id = list(spatial_mapping.keys())[0]
            else:
                library_id = None
        if library_id is not None:
            spatial_data = spatial_mapping[library_id]
        else:
            spatial_data = None
        return library_id, spatial_data


    def main(name:str):
        print(f'Hello {name}')

if __name__ == "__main__":
    k=Kepler()
