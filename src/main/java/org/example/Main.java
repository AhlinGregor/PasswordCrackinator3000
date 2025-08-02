package org.example;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");

        switch (args[0]) {
            case "1":
                GrafikaSeq.createAndShowGUI();
                break;
            case "2":
                GrafikaMulti.createAndShowGUI();
                break;
            case "3":
                GrafikaCUDA.createAndShowGUI();
                break;
        }
        // GrafikaSeq.createAndShowGUI();
        // GrafikaMulti.createAndShowGUI();
        // GrafikaCUDA.createAndShowGUI();

    }
}