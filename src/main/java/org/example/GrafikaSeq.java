package org.example;

import javax.swing.*;
import java.awt.*;
import java.io.File;

public class GrafikaSeq {
    public static void createAndShowGUI() {
        // Create the main frame
        JFrame frame = new JFrame("GrafikaSeq GUI");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(500, 200);

        // Create a panel to hold all components
        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(5, 2)); // 6 rows for the components

        // Add a text input field
        JTextField textField = new JTextField(20);
        panel.add(new JLabel("Enter text:"));
        panel.add(textField);

        // Add radio buttons
        JRadioButton md5 = new JRadioButton("MD-5");
        JRadioButton sha = new JRadioButton("SHA-256");
        ButtonGroup radioGroup = new ButtonGroup(); // Group the radio buttons
        radioGroup.add(md5);
        radioGroup.add(sha);

        JPanel radioPanel = new JPanel();
        radioPanel.add(md5);
        radioPanel.add(sha);
        panel.add(new JLabel("Select an option:"));
        panel.add(radioPanel);

        // Add checkboxes
        JCheckBox small = new JCheckBox("a-z");
        JCheckBox big = new JCheckBox("A-Z");
        JCheckBox nonAlpha = new JCheckBox("special");

        JPanel checkboxPanel = new JPanel();
        checkboxPanel.add(small);
        checkboxPanel.add(big);
        checkboxPanel.add(nonAlpha);
        panel.add(new JLabel("Select options:"));
        panel.add(checkboxPanel);

        //add length chooser
        SpinnerNumberModel model = new SpinnerNumberModel(5, 1, 16, 1);
        JSpinner spinner = new JSpinner(model);

        panel.add(new JLabel("Length:"));
        panel.add(spinner);

        // Add a progres bar
        JProgressBar progressBar = new JProgressBar(0, 100);
        progressBar.setValue(0);
        progressBar.setStringPainted(true);

        // Add a button
        JButton button = new JButton("Brute force");
        button.addActionListener(e -> {
            progressBar.setValue(0); // Reset progress bar

            new Thread(() -> {
                String hash = textField.getText();
                boolean md5Selected = md5.isSelected();
                int selectedCheckboxes = (small.isSelected() ? 1 : 0) + (big.isSelected() ? 2 : 0) + (nonAlpha.isSelected() ? 4 : 0);
                int selectedInteger = (int) spinner.getValue();

                // Calculate total combinations
                String available = SequentialSolution.getCharacterSet(selectedCheckboxes);
                long totalCombinations = SequentialSolution.calculateTotalCombinations(available.length(), selectedInteger);

                // Update progress bar's text format
                progressBar.setStringPainted(true);
                progressBar.setMaximum((int) totalCombinations);

                long start = System.currentTimeMillis();
                String result = SequentialSolution.computeDizShiz(
                        hash,
                        md5Selected,
                        selectedCheckboxes,
                        selectedInteger,
                        progressBar,
                        totalCombinations
                );
                long stop = System.currentTimeMillis();

                // Show the result in a new window
                SwingUtilities.invokeLater(() -> {
                    if (result != null) {
                        JOptionPane.showMessageDialog(
                                frame,
                                "Solution Found: " + result + "\n Time to crack: " + (stop-start) + "ms",
                                "Result",
                                JOptionPane.INFORMATION_MESSAGE
                        );
                    } else {
                        JOptionPane.showMessageDialog(
                                frame,
                                "No solution was found.",
                                "Result",
                                JOptionPane.WARNING_MESSAGE
                        );
                    }
                });
            }).start();
        });
        panel.add(button);

        JButton chooseFile = new JButton("Dictionary");
        chooseFile.addActionListener(e -> {
            // Get text from the text field
            String hash = textField.getText();

            // Get selected radio button
            boolean md5Selected = md5.isSelected();

            JFileChooser fileChooser = new JFileChooser();
            int result = fileChooser.showOpenDialog(panel);

            if (result == JFileChooser.APPROVE_OPTION) {
                File file = fileChooser.getSelectedFile();

                // Check if it's a .txt file
                if (file.getName().endsWith(".txt")) {
                    long start = System.currentTimeMillis();
                    String resitev = SequentialSolution.dictionaryAttack(
                            file,
                            hash,
                            md5Selected
                    );
                    long stop = System.currentTimeMillis();

                    // Show the result in a new window
                    SwingUtilities.invokeLater(() -> {
                        if (resitev != null) {
                            JOptionPane.showMessageDialog(
                                    frame,
                                    "Solution Found: " + resitev + "\n Time to crack: " + (stop-start) + "ms",
                                    "Result",
                                    JOptionPane.INFORMATION_MESSAGE
                            );
                        } else {
                            JOptionPane.showMessageDialog(
                                    frame,
                                    "No solution was found.",
                                    "Result",
                                    JOptionPane.WARNING_MESSAGE
                            );
                        }
                    });
                } else {
                    JOptionPane.showMessageDialog(frame, "Please select a .txt file", "Invalid File", JOptionPane.ERROR_MESSAGE);
                }
            }
        });
        panel.add(chooseFile);

        frame.add(progressBar, BorderLayout.SOUTH);

        // Add the panel to the frame
        frame.add(panel);

        // Set the frame to be visible
        frame.setVisible(true);
    }
}
