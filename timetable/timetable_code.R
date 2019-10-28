library(timevis)

data <- data.frame(
  id      = 1:4,
  content = c("Item one", "Item two",
              "Ranged item", "Item four"),
  start   = c("2019-10-01", "2016-01-11",
              "2016-01-20", "2016-02-14 15:00:00"),
  end     = c("2020-04-14", NA, "2016-02-04", NA)
)

timevis(data,width = 500, height = 200)
