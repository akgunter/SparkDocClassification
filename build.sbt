val sparkVersion = "2.3.1"

lazy val scalaclassifier =
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
        "com.github.scopt" %% "scopt" % "3.7.0",
        "org.scalaz" %% "scalaz-core" % "7.2.23",
        "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
        ("org.deeplearning4j" %% "dl4j-spark" % "1.0.0-beta_spark_2")
          .exclude("commons-beanutils", "commons-beanutils")
          .exclude("commons-collections", "commons-collections")
          .exclude("org.apache.hadoop", "hadoop-yarn-api")
          .exclude("org.glassfish.hk2.external", "aopalliance-repackaged")
          .exclude("org.glassfish.hk2.external", "javax.inject")
          .exclude("org.slf4j", "jcl-over-slf4j")
          .exclude("org.slf4j", "slf4j-log4j12")
          .exclude("org.jetbrains", "annotations"),
        ("org.datavec" % "datavec-spark_2.11" % "1.0.0-beta_spark_2")
          .exclude("commons-beanutils", "commons-beanutils")
          .exclude("commons-collections", "commons-collections")
          .exclude("org.apache.hadoop", "hadoop-yarn-api")
          .exclude("org.codehaus.janino", "janino")
          .exclude("org.glassfish.hk2.external", "aopalliance-repackaged")
          .exclude("org.glassfish.hk2.external", "javax.inject"),
        ("org.deeplearning4j" %% "dl4j-spark-parameterserver" % "1.0.0-beta_spark_2")
          .exclude("org.apache.tomcat", "tomcat-servlet-api")
          .exclude("io.aeron", "aeron-all")
          .exclude("commons-logging", "commons-logging")
          .exclude("org.iq80.leveldb", "leveldb-api")
          .exclude("org.jetbrains", "annotations")
          .exclude("org.slf4j", "jcl-over-slf4j")
          .exclude("org.slf4j", "slf4j-log4j12")
      )
    )
