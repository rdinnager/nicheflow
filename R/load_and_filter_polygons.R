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
  areas <- map(polys$geometry, st_area, .progress = TRUE)
  areas <- list_c(areas)
  polys <- polys |>
    filter(areas > 0.01)
  polys |>
    select(binomial, family, area, geometry)
}
