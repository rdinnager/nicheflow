#' .. content for \description{} (no empty lines) ..
#'
#' .. content for \details{} ..
#'
#' @title
#' @param nameme1
#' @return
#' @author rdinnage
#' @export
load_and_filter_polygons <- function(folder = "data/SDM/maps/GARD1.7") {

  polys <- st_read(folder)
  
  invalid <- map_lgl(polys$geometry, st_is_valid, .progress = TRUE)
  polys$geometry[!invalid] <- st_make_valid(polys$geometry[!invalid])
  invalid2 <- map_lgl(polys$geometry[!invalid], st_is_valid, .progress = TRUE)
  exclude <- which(!invalid)[!invalid2]
  polys <- polys |>
    slice(-exclude)
  #invalid <- map_lgl(polys$geometry, st_is_valid, .progress = TRUE)
  ## remove very small range species and species with very difficultly
  ## shaped ranges (e.g. lots of small islands or very long and skinny)
  areas <- map(1:nrow(polys), 
               ~ possibly(st_area, 0)(polys[.x, ]), 
               .progress = TRUE)
  areas <- map_dbl(areas, possibly(as.numeric, 0))
  areas <- areas / 1e6
  plot(polys[areas > 100 & areas < 200, ]$geometry[[1]])

  bb_area <- map(1:nrow(polys), 
                 possibly(~ st_bbox(polys[.x, ]) |>
                            st_as_sfc() |>
                            st_area(), 0), 
                 .progress = TRUE)
  bb_areas <- map_dbl(bb_area, possibly(as.numeric, 0))
  bb_areas <- bb_areas / 1e6
  area_ratios <- areas/bb_areas
  hist(area_ratios[area_ratios < 1], breaks = 100)
  plot(polys[area_ratios > 0.1 & area_ratios < 0.11, ]$geometry[[20]])
  plot(polys[area_ratios > 1, ]$geometry[[4]])
  
  circles <- abs(area_ratios - (pi/4)) < 0.00015
  very_small <- areas < 100
  weird_shapes <- area_ratios < 0.01
  remove <- circles | very_small | weird_shapes
  
  polys <- polys |>
    dplyr::filter(!remove)
  ## manually remove some problematic ranges
  polys <- polys |>
    dplyr::filter(!binomial %in% c("Emoia impar", "Emoia cyanura", "Emoia trossula"))
  
  polys |>
    select(binomial, family, area, geometry)
}
