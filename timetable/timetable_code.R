library(readxl)
library(timevis)

timeline_table <- read_excel("timeline_table.xlsx")


data <- data.frame(
  id      = 1:nrow(timeline_table),
  content = timeline_table$content,
  start   = timeline_table$start,
  end     = timeline_table$end
)

timevis(data,width = 1500, height = 200)

x = c(0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25)
y = c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9, 1)
plot(x, y, col='red', xlab = "complexity", ylab = "#paras", xaxt='n', yaxt='n', 
     xlim=c(0, 1), ylim = c(0,1), pch =19)
points(0.75, 0.1, col="blue", pch = 19)
points(0.75, 0.2, col = 'green', pch = 19)
points(0.8, 0.5, col = 'black', pch = 19)
points(1, 1, col = 'grey', pch = 19)
legend(x = c(0.28,0.75), y = c(0.6, 1.1), col= c('red', 'blue', 'green', "black"), legend = c("linear model(model 1)", 'batch reaction(model 2)', 'urine_nitrification(model 3)', 'hydrological model(model 4)'), 
       pch = 19, cex = 0.8, pt.cex= 1)

