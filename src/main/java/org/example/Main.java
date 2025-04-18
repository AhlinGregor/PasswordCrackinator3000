package org.example;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        /*
        switch (args[1]) {
            case "1":
                sequentialSollution();
                break;
            case "2":
                parallelSollution();
                break;
            case "3":
                cudaSollution();
                break;
        }*/
        // GrafikaSeq.createAndShowGUI();
        GrafikaMulti.createAndShowGUI();

    }
/*
    public static void sequentialSollution() {
        GrafikaSeq.createAndShowGUI();
    }
    public static void parallelSollution() {}
    public static void cudaSollution() {}*/
}