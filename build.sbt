val sparkVersion = "2.3.1"

lazy val sparklearning =
  (project in file(".")).
    settings(
      name := "SparkLearning",
      organization := "net.ddns.akgunter",
      version := "0.1",
      scalaVersion := "2.11.12",
      test in assembly := {},
      assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false),
      assemblyMergeStrategy in assembly := {
        case PathList("log4j.properties") => MergeStrategy.discard
        case PathList("log4j.xml") => MergeStrategy.discard
        case PathList(xs @ _*) if xs.last == "UnusedStubClass.class" =>
          MergeStrategy.first
        case PathList(xs @ _*) if xs.last == "overview.html" =>
          MergeStrategy.first
        case x =>
          val oldStrategy = (assemblyMergeStrategy in assembly).value
          oldStrategy(x)
      },
      libraryDependencies ++= Seq(
        "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
        "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta",
        ("org.deeplearning4j" %% "dl4j-spark" % "1.0.0-beta_spark_2")
          .exclude("commons-beanutils", "commons-beanutils")
          .exclude("commons-collections", "commons-collections")
          .exclude("org.apache.hadoop", "hadoop-yarn-api")
          .exclude("org.glassfish.hk2.external", "aopalliance-repackaged")
          .exclude("org.glassfish.hk2.external", "javax.inject")
          .exclude("org.slf4j", "slf4j-log4j12")
          .exclude("org.slf4j", "jcl-over-slf4j")
          .exclude("org.jetbrains", "annotations")
          .exclude("org.bytedeco.javacpp-presets", "opencv"),
        ("org.deeplearning4j" %% "dl4j-spark-parameterserver" % "1.0.0-beta_spark_2")
          .exclude("org.apache.tomcat", "tomcat-servlet-api")
          .exclude("org.agrona", "Agrona")
          .exclude("commons-logging", "commons-logging")
          .exclude("org.iq80.leveldb", "leveldb-api"),
        "org.deeplearning4j" %% "scalnet" % "1.0.0-beta",
        ("org.datavec" % "datavec-spark_2.11" % "1.0.0-beta_spark_2")
          .exclude("commons-beanutils", "commons-beanutils")
          .exclude("commons-collections", "commons-collections")
          .exclude("org.apache.hadoop", "hadoop-yarn-api")
          .exclude("org.codehaus.janino", "janino")
          .exclude("org.glassfish.hk2.external", "aopalliance-repackaged")
          .exclude("org.glassfish.hk2.external", "javax.inject")
          .exclude("org.slf4j", "slf4j-log4j12")
      )
    )
