lazy val scalaclassifier =
  (project in file(".")).
    settings(
      name := "ScalaClassifier",
      organization := "net.ddns.akgunter",
      version := "0.1",
      scalaVersion := "2.12.6",
      test in assembly := {},
      parallelExecution in Test := false
    )
