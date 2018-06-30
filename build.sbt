val sparkVersion = "2.3.1"

lazy val scalaclassifier =
  (project in file(".")).
    settings(
      name := "ScalaClassifier",
      organization := "net.ddns.akgunter",
      version := "0.1",
      scalaVersion := "2.11.12",
      test in assembly := {},
      assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false),
      assemblyMergeStrategy in assembly := {
        case PathList("log4j.properties") => MergeStrategy.discard
        case PathList("log4j.xml") => MergeStrategy.discard
        case PathList("org.deeplearning4j.spark") => MergeStrategy.discard
        case PathList("org.datavec.spark") => MergeStrategy.discard
        case x =>
          val oldStrategy = (assemblyMergeStrategy in assembly).value
          oldStrategy(x)
      },
      libraryDependencies ++= Seq(
        "com.github.scopt" %% "scopt" % "3.7.0",
        "org.scalaz" %% "scalaz-core" % "7.2.23",
        "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
        "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-beta_spark_2",
        "org.datavec" % "datavec-spark_2.11" % "1.0.0-beta_spark_2"
      )
    )
