#  R Test
checkPrime <- function(num) {
  if (num <= 2) {
    FALSE
  } else if (any(num %% 2:(num-1) == 0)) {
    FALSE
  } else { 
    TRUE
  }
}


checkPrime(523)
nvec<- 3:1000
primes<-sapply(nvec,checkPrime)
primes[10]


proportion <- cumsum(primes)/nvec
plot(nvec, proportion, xlab="n",ylab="prime proportion",type="l")


nvec <- 3:10000
primes <- sapply(nvec,checkPrime)
proportion <- cumsum(primes)/nvec
plot(nvec, proportion, xlab="n",ylab="prime proportion",type="l")




linerModel <- lm(Fertility ~ . , data = swiss)
summary(linerModel)

nulleModel <- lm(Fertility ~ 1,data = swiss)  
anova(nulleModel,linerModel)

par(mfrow = c(2,2))
plot(linerModel)


aic_Model <-  step(linerModel)

#3
#a
motorins <- read.csv("motorins.csv")
motorins$Make <- factor(motorins$Make)
model_glm <- glm(Claims~ Make + Insured,
                 family="poisson",data=motorins)

model_glm$coefficients[1]+model_glm$coefficients[2:9]

#b
model_glm_intr <- glm(Claims ~ Make * Insured,
               family="poisson",data=motorins)


model_glm_intr$coefficients[1]+model_glm_intr$coefficients[2:9]
#c
anova(model_glm,model_glm_intr)
pchisq(22162,8,lower.tail = F)

#d
mu <- mean(motorins$Insured)
predict(model_glm, newdata=data.frame(Make=factor(8),Insured=mu),type="response")
predict(model_glm_intr, newdata=data.frame(Make=factor(8),Insured=mu),type="response")
lp <- model_glm$coefficients[1]+model_glm$coefficients[8]+model_glm$coefficients[10]*mu

(lp1 <- model_glm_intr$coefficients[1]+model_glm_intr$coefficients[8]
  +model_glm_intr$coefficients[10]*mu+model_glm_intr$coefficients[17]*mu)

(mean <- exp(lp))

(mean <- exp(lp1))