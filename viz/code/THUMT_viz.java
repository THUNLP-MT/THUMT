import java.util.List;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import javax.swing.*;

public class THUMT_Viz {
	
	private MenuBar bar;
	private Menu filemenu;
	private MenuItem open;
	private FileDialog openDia;
	public int fileIndex = 0;
	public List<BufferedReader> readerList = new ArrayList<BufferedReader>();
	private int Lensrc = 0;
	private int Lentrg = 0;
	private PaintFrame PF;

	private THUMT_Viz() throws IOException{
		bar = new MenuBar();
		filemenu = new Menu("File");
		open = new MenuItem("Open ...");
		filemenu.add(open);
		bar.add(filemenu);
		PF = new PaintFrame( "THUMT_Viz" );
		openDia = new FileDialog(PF, "Open", FileDialog.LOAD);
		PF.setMenuBar(bar);
		try{
			init initStr= new init();
			String Str = initStr.str;
			
			String[] strList = Str.split("\\|\\|\\|");
			//System.out.println("fff");
			int idx = 0;
			String line = null;
			line = strList[idx];
			//System.out.println(line);
			idx ++;
		    PF.srcList = line.split(" ");
			line = strList[idx];
			//System.out.println(line);
			idx ++;
			PF.trgList = line.split(" ");
			Lensrc = PF.srcList.length;
			Lentrg = PF.trgList.length;
			PF.R_enc_x = new float[Lensrc][Lensrc];
			PF.R_enc_x_f = new float[Lensrc][Lensrc];
			PF.R_enc_x_b = new float[Lensrc][Lensrc];
			PF.R_ctx_x = new float[Lentrg][Lensrc];
			PF.R_dec_x = new float[Lentrg][Lensrc];
			PF.R_dec_y = new float[Lentrg][Lentrg];
			PF.probs = new float[Lentrg][Lensrc];
			PF.trg_x = new float[Lentrg][Lensrc];
			PF.trg_y = new float[Lentrg][Lentrg];
			
			
			PF.R_enc_x_y = new float[Lensrc][Lensrc];
			PF.R_enc_x_f_y = new float[Lensrc][Lensrc];
			PF.R_enc_x_b_y = new float[Lensrc][Lensrc];
			PF.R_ctx_x_y = new float[Lentrg][Lensrc];
			PF.R_dec_x_y = new float[Lentrg][Lensrc];
			PF.R_dec_y_y = new float[Lentrg][Lentrg];
			PF.probs_y = new float[Lentrg][Lensrc];
			PF.trg_x_y = new float[Lentrg][Lensrc];
			PF.trg_y_y = new float[Lentrg][Lentrg];
			
			PF.R_enc_x_s = new float[Lensrc][Lensrc];
			PF.R_enc_x_f_s = new float[Lensrc][Lensrc];
			PF.R_enc_x_b_s = new float[Lensrc][Lensrc];
			PF.R_ctx_x_s = new float[Lentrg][Lensrc];
			PF.R_dec_x_s = new float[Lentrg][Lensrc];
			PF.R_dec_y_s = new float[Lentrg][Lentrg];
			PF.probs_s = new float[Lentrg][Lensrc];
			PF.trg_x_s = new float[Lentrg][Lensrc];
			PF.trg_y_s = new float[Lentrg][Lentrg];
			
			for(int i = 0; i < Lensrc; i ++){
				float max_j  = -1;
				line = strList[idx];
				idx ++;
				String [] tmp = line.split(" ");
				for(int j = 0;j < Lensrc;j ++){
					
					PF.R_enc_x_f_y[i][j] = Float.parseFloat(tmp[j]);  
					if(PF.R_enc_x_f_y[i][j] > max_j)
						max_j = PF.R_enc_x_f_y[i][j];
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_enc_x_f[i][j] = PF.R_enc_x_f_y[i][j] / max_j;
					sum += PF.R_enc_x_f[i][j];
				}
				for(int j = 0; j < Lensrc;j ++)
					PF.R_enc_x_f_s[i][j] = PF.R_enc_x_f[i][j]/sum;
			}
			
			for(int i = 0; i < Lensrc; i ++){
				float max_j  = -1;
				line = strList[idx];
				idx ++;
				String [] tmp = line.split(" ");
				for(int j = 0; j < Lensrc; j ++){
					PF.R_enc_x_b_y[i][j] = Float.parseFloat(tmp[j]); 
					if(PF.R_enc_x_b_y[i][j] > max_j)
						max_j = PF.R_enc_x_b_y[i][j];
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_enc_x_b[i][j] = PF.R_enc_x_b_y[i][j] / max_j;
					sum += PF.R_enc_x_b[i][j];
				}
				for(int j = 0; j < Lensrc;j ++)
					PF.R_enc_x_b_s[i][j] = PF.R_enc_x_b[i][j]/sum;
			}
			
			for(int i = 0; i < Lensrc; i ++){
				float max_j  = -1;
				line = strList[idx];
				idx ++;
				String [] tmp = line.split(" ");
				for(int j = 0; j < Lensrc; j ++){
					PF.R_enc_x_y[i][j] = Float.parseFloat(tmp[j]); 
					if(PF.R_enc_x_y[i][j] > max_j){
						max_j = PF.R_enc_x_y[i][j];
					}
					
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_enc_x[i][j] = PF.R_enc_x_y[i][j] / max_j;
					sum += PF.R_enc_x[i][j];
				}	
				for(int j = 0; j < Lensrc;j ++)
					PF.R_enc_x_s[i][j] = PF.R_enc_x[i][j]/sum;
			}
			
			for (int i = 0; i < Lentrg; i ++){
				float max_j  = -1;
				line = strList[idx];
				idx ++;
				String [] tmp = line.split(" ");
				for (int j = 0; j < Lensrc; j ++){
					PF.R_ctx_x_y[i][j] = Float.parseFloat(tmp[j]);
					if(PF.R_ctx_x_y[i][j] > max_j){
						max_j = PF.R_ctx_x_y[i][j];
					}
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_ctx_x[i][j] = PF.R_ctx_x_y[i][j] / max_j;
					sum += PF.R_ctx_x[i][j];
				}
				for(int j = 0; j < Lensrc;j ++)
					PF.R_ctx_x_s[i][j] = PF.R_ctx_x[i][j]/sum;
			}
			
			for(int i = 0; i < Lentrg; i ++){
				float max_j  = -1;
				line = strList[idx];
				idx ++;
				String [] tmp = line.split(" ");
				for(int j = 0; j < Lensrc; j ++){
					PF.R_dec_x_y[i][j] = Float.parseFloat(tmp[j]); 
					if(PF.R_dec_x_y[i][j] > max_j){
						max_j = PF.R_dec_x_y[i][j];
					}
				}
				//float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_dec_x[i][j] = PF.R_dec_x_y[i][j] / max_j;
				}
			}
			

			for(int i = 0; i < Lentrg; i ++){
				float max_j  = -1;
				line = strList[idx];
				idx ++;
				String [] tmp = line.split(" ");
				for(int j = 0; j < Lentrg; j ++){
					PF.R_dec_y_y[i][j] = Float.parseFloat(tmp[j]); 
					if(PF.R_dec_y_y[i][j] > max_j){
						max_j = PF.R_dec_y_y[i][j];
					}
				}
				//float sum = 0;
				for(int j = 0; j< Lentrg;j ++){
					PF.R_dec_y[i][j] = PF.R_dec_y_y[i][j] / max_j;
				}
			}
			for(int i = 0;i < Lentrg; i ++){
				float max_j = -1;
				
				for (int j = 0;j < Lensrc; j++){
					if(PF.R_dec_x_y[i][j] > max_j)
						max_j = PF.R_dec_x_y[i][j];
				}
				for (int j = 0;j < Lentrg; j++){
					if(PF.R_dec_y_y[i][j] > max_j)
						max_j = PF.R_dec_y_y[i][j];
				}
				float sum  = 0;
				for(int j = 0;j < Lensrc;j++){
					PF.R_dec_x[i][j] = PF.R_dec_x_y[i][j] / max_j;
					sum += PF.R_dec_x[i][j];
				}
				for( int j = 0;j < Lentrg;j ++){
					PF.R_dec_y[i][j] = PF.R_dec_y_y[i][j] / max_j;
					sum += PF.R_dec_y[i][j];
				}
				for(int j = 0;j < Lensrc;j++){
					PF.R_dec_x_s[i][j] = PF.R_dec_x[i][j] / sum;
					//sum += PF.R_dec_x[i][j];
				}
				for( int j = 0;j < Lentrg;j ++){
					PF.R_dec_y_s[i][j] = PF.R_dec_y[i][j] / sum;
					//sum += PF.R_dec_y[i][j];
				}
			}
			for(int i = 0; i < Lentrg; i ++){
				float max_j  = -1;
				line = strList[idx];
				idx ++;
				String [] tmp = line.split(" ");
				for(int j = 0; j < Lensrc; j ++){
					PF.probs_y[i][j] = Float.parseFloat(tmp[j]); 	
					if(PF.probs_y[i][j] > max_j){
						max_j = PF.probs_y[i][j];
					}
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.probs[i][j] = PF.probs_y[i][j] / max_j;
					sum += PF.probs[i][j];
				}
				for(int j = 0; j< Lensrc;j ++){
					PF.probs_s[i][j] = PF.probs[i][j] / sum;
				}
			}
			
			for(int i = 0; i < Lentrg; i ++){
				float max_j  = -1;
				line = strList[idx];
				idx ++;
				String [] tmp = line.split(" ");
				for(int j = 0; j < Lensrc; j ++){
					PF.trg_x_y[i][j] = Float.parseFloat(tmp[j]);
					if (PF.trg_x_y[i][j] > max_j){
						max_j = PF.trg_x_y[i][j];
					}
				}

				for(int j = 0; j < Lensrc; j ++){
					PF.trg_x[i][j] = PF.trg_x_y[i][j] / max_j;
				}
				
			}
			
			for(int i = 0; i < Lentrg; i ++){
				float max_j  = -1;
				line = strList[idx];
				idx ++;
				String [] tmp = line.split(" ");
				for(int j = 0;j < Lentrg; j ++){
					PF.trg_y_y [i][j] = Float.parseFloat(tmp[j]);
					if (PF.trg_y_y[i][j] > max_j){
						max_j = PF.trg_y_y[i][j];
					}
				}
				
				for(int j = 0; j< Lentrg; j ++){
					PF.trg_y[i][j] = PF.trg_y_y[i][j] / max_j ;					
				}
                
            }
			
			for(int i = 0;i < Lentrg; i ++){
				float max_j = -1;
				for (int j = 0;j < Lensrc; j++){
					if(PF.trg_x_y[i][j] > max_j)
						max_j = PF.trg_x_y[i][j];
				}
				for (int j = 0;j < Lentrg; j++){
					if(PF.trg_y_y[i][j] > max_j)
						max_j = PF.trg_y_y[i][j];
				}
				float sum = 0;
				for(int j = 0;j < Lensrc;j++){
					PF.trg_x[i][j] = PF.trg_x_y[i][j] / max_j;
					sum += PF.trg_x[i][j];
				}
				for( int j = 0;j < Lentrg;j ++){
					PF.trg_y[i][j] = PF.trg_y_y[i][j] / max_j;
					sum += PF.trg_y[i][j];
				}
				for(int j = 0;j < Lensrc;j++){
					PF.trg_x_s[i][j] = PF.trg_x[i][j] / sum;
					//sum += PF.trg_x[i][j];
				}
				for( int j = 0;j < Lentrg;j ++){
					PF.trg_y_s[i][j] = PF.trg_y[i][j] / sum;
					//sum += PF.trg_y[i][j];
				}
			}
			
		}catch(Exception e){
			System.out.println("error: " + e.toString());
			e.printStackTrace();
		}
		PF.jp.repaint();
		MyEvent();
    }

	private void MyEvent(){
		
		open.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				openDia.setVisible(true);
				String dirPath = openDia.getDirectory();
				String fileName = openDia.getFile();
				if(dirPath == null || fileName == null)
					return;
				fileName = dirPath + fileName;
				
				try {
					readerList.clear();
					int num = 0;
					ZipFile zipfile = new ZipFile(fileName);
					InputStream in = new BufferedInputStream(new FileInputStream(fileName));
					ZipInputStream zin = new ZipInputStream(in);
					ZipEntry ze;
					while((ze = zin.getNextEntry()) != null){
						if(ze.isDirectory()){
							
						}
						else{
							if(ze.getName().contains(".txt") && ! ze.getName().contains("__MACOSX")){
								BufferedReader bufr = new BufferedReader(new InputStreamReader(zipfile.getInputStream(ze)));
								bufr.mark(1000000);
								readerList.add(bufr);
								num ++;
							}
						}		
					}
					PF.total = num;
					PF.totalLabel.setText("/   " + Integer.toString(PF.total));
			} catch (IOException e3) {	
				System.out.println("error: " + e3.toString());
				e3.printStackTrace();
			}
				BufferedReader br = null;
				readData(0, br);
				fileIndex = 0;
            }
		});
		
		PF.jtf.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				if(e.getSource() == PF.jtf){
					fileIndex = Integer.parseInt(PF.jtf.getText());
					if(readerList.size() == 0){
						fileIndex = 0;
					}
					else{
						if(fileIndex >= readerList.size()){
							fileIndex = readerList.size() -1;
							PF.jtf.setText(Integer.toString(fileIndex + 1));
						}
						if(fileIndex < 0){
							fileIndex = 0;
							PF.jtf.setText(Integer.toString(fileIndex + 1));
						}
						BufferedReader br = null;
						readData(fileIndex, br);
					}
				}
			}
		});
		
		PF.jb1.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				fileIndex ++;
				if(fileIndex >= readerList.size())
					fileIndex --;
				//System.out.println("file Index" + fileIndex);
				BufferedReader br = null;
				PF.jtf.setText(Integer.toString(fileIndex + 1));
				readData(fileIndex, br);
			}
		});
		
		PF.jb2.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				fileIndex --;
				if(fileIndex < 0)
					fileIndex = 0;
				PF.jtf.setText(Integer.toString(fileIndex + 1));
				BufferedReader br = null;
				readData(fileIndex, br);
			}
		});
		
		PF.jb3.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				fileIndex += 10;
				if(fileIndex >= readerList.size())
					fileIndex = readerList.size() - 1;
				if(readerList.size() == 0)
					fileIndex = 0;
				PF.jtf.setText(Integer.toString(fileIndex + 1));
				BufferedReader br = null;
				readData(fileIndex, br);
			}
		});
		
		PF.jb4.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				fileIndex -= 10;
				if(fileIndex < 0)
					fileIndex = 0;
				PF.jtf.setText(Integer.toString(fileIndex + 1));
				BufferedReader br = null;
				readData(fileIndex, br);
			}
		});
	}
	
	public void readData(int index,BufferedReader br){
		try{
			
			BufferedReader bufr;
			System.out.println("index " + index);
			if(index >= 0 )
				bufr = readerList.get(index);
			else 
				bufr = br;
			PF.fileName = "";
			String line = null;
			line = bufr.readLine().trim();
		    PF.srcList = line.split(" ");
			line = bufr.readLine().trim();
			PF.trgList = line.split(" ");
			Lensrc = PF.srcList.length;
			Lentrg = PF.trgList.length;
			PF.R_enc_x = new float[Lensrc][Lensrc];
			PF.R_enc_x_f = new float[Lensrc][Lensrc];
			PF.R_enc_x_b = new float[Lensrc][Lensrc];
			PF.R_ctx_x = new float[Lentrg][Lensrc];
			PF.R_dec_x = new float[Lentrg][Lensrc];
			PF.R_dec_y = new float[Lentrg][Lentrg];
			PF.probs = new float[Lentrg][Lensrc];
			PF.trg_x = new float[Lentrg][Lensrc];
			PF.trg_y = new float[Lentrg][Lentrg];
			
			PF.R_enc_x_y = new float[Lensrc][Lensrc];
			PF.R_enc_x_f_y = new float[Lensrc][Lensrc];
			PF.R_enc_x_b_y = new float[Lensrc][Lensrc];
			PF.R_ctx_x_y = new float[Lentrg][Lensrc];
			PF.R_dec_x_y = new float[Lentrg][Lensrc];
			PF.R_dec_y_y = new float[Lentrg][Lentrg];
			PF.probs_y = new float[Lentrg][Lensrc];
			PF.trg_x_y = new float[Lentrg][Lensrc];
			PF.trg_y_y = new float[Lentrg][Lentrg];
			
			PF.R_enc_x_s = new float[Lensrc][Lensrc];
			PF.R_enc_x_f_s = new float[Lensrc][Lensrc];
			PF.R_enc_x_b_s = new float[Lensrc][Lensrc];
			PF.R_ctx_x_s = new float[Lentrg][Lensrc];
			PF.R_dec_x_s = new float[Lentrg][Lensrc];
			PF.R_dec_y_s = new float[Lentrg][Lentrg];
			PF.probs_s = new float[Lentrg][Lensrc];
			PF.trg_x_s = new float[Lentrg][Lensrc];
			PF.trg_y_s = new float[Lentrg][Lentrg];
			
			for(int i = 0; i < Lensrc; i ++){
				float max_j  = -1;
				line = bufr.readLine().trim();
				String [] tmp = line.split(" ");
				for(int j = 0;j < Lensrc;j ++){
					
					PF.R_enc_x_f_y[i][j] = Float.parseFloat(tmp[j]);  
					if(PF.R_enc_x_f_y[i][j] > max_j)
						max_j = PF.R_enc_x_f_y[i][j];
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_enc_x_f[i][j] = PF.R_enc_x_f_y[i][j] / max_j;
					sum += PF.R_enc_x_f[i][j];
				}
				for(int j = 0; j < Lensrc;j ++)
					PF.R_enc_x_f_s[i][j] = PF.R_enc_x_f[i][j]/sum;
			}
			
			for(int i = 0; i < Lensrc; i ++){
				line = bufr.readLine().trim();
				float max_j  = -1;
				String [] tmp = line.split(" ");
				for(int j = 0; j < Lensrc; j ++){
					PF.R_enc_x_b_y[i][j] = Float.parseFloat(tmp[j]); 
					if(PF.R_enc_x_b_y[i][j] > max_j)
						max_j = PF.R_enc_x_b_y[i][j];
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_enc_x_b[i][j] = PF.R_enc_x_b_y[i][j] / max_j;
					sum += PF.R_enc_x_b[i][j];
				}
				for(int j = 0; j < Lensrc;j ++)
					PF.R_enc_x_b_s[i][j] = PF.R_enc_x_b[i][j]/sum;
			}
			
			for(int i = 0; i < Lensrc; i ++){
				line = bufr.readLine().trim();
				float max_j  = -1;
				String [] tmp = line.split(" ");
				for(int j = 0; j < Lensrc; j ++){
					PF.R_enc_x_y[i][j] = Float.parseFloat(tmp[j]); 
					if(PF.R_enc_x_y[i][j] > max_j){
						max_j = PF.R_enc_x_y[i][j];
					}
					
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_enc_x[i][j] = PF.R_enc_x_y[i][j] / max_j;
					sum += PF.R_enc_x[i][j];
				}	
				for(int j = 0; j < Lensrc;j ++)
					PF.R_enc_x_s[i][j] = PF.R_enc_x[i][j]/sum;
			}
			
			for (int i = 0; i < Lentrg; i ++){
				line = bufr.readLine().trim();
				float max_j  = -1;
				String [] tmp = line.split(" ");
				for (int j = 0; j < Lensrc; j ++){
					PF.R_ctx_x_y[i][j] = Float.parseFloat(tmp[j]);
					if(PF.R_ctx_x_y[i][j] > max_j){
						max_j = PF.R_ctx_x_y[i][j];
					}
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_ctx_x[i][j] = PF.R_ctx_x_y[i][j] / max_j;
					sum += PF.R_ctx_x[i][j];
				}
				for(int j = 0; j < Lensrc;j ++)
					PF.R_ctx_x_s[i][j] = PF.R_ctx_x[i][j]/sum;
			}
			
			for(int i = 0; i < Lentrg; i ++){
				line = bufr.readLine().trim();
				String [] tmp = line.split(" ");
				float max_j  = -1;
				for(int j = 0; j < Lensrc; j ++){
					PF.R_dec_x_y[i][j] = Float.parseFloat(tmp[j]); 
					if(PF.R_dec_x_y[i][j] > max_j){
						max_j = PF.R_dec_x_y[i][j];
					}
				}
				//float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.R_dec_x[i][j] = PF.R_dec_x_y[i][j] / max_j;
				}
			}
			

			for(int i = 0; i < Lentrg; i ++){
				line = bufr.readLine().trim();
				String [] tmp = line.split(" ");
				float max_j  = -1;
				for(int j = 0; j < Lentrg; j ++){
					PF.R_dec_y_y[i][j] = Float.parseFloat(tmp[j]); 
					if(PF.R_dec_y_y[i][j] > max_j){
						max_j = PF.R_dec_y_y[i][j];
					}
				}
				//float sum = 0;
				for(int j = 0; j< Lentrg;j ++){
					PF.R_dec_y[i][j] = PF.R_dec_y_y[i][j] / max_j;
				}
			}
			for(int i = 0;i < Lentrg; i ++){
				float max_j = -1;
				
				for (int j = 0;j < Lensrc; j++){
					if(PF.R_dec_x_y[i][j] > max_j)
						max_j = PF.R_dec_x_y[i][j];
				}
				for (int j = 0;j < Lentrg; j++){
					if(PF.R_dec_y_y[i][j] > max_j)
						max_j = PF.R_dec_y_y[i][j];
				}
				float sum  = 0;
				for(int j = 0;j < Lensrc;j++){
					PF.R_dec_x[i][j] = PF.R_dec_x_y[i][j] / max_j;
					sum += PF.R_dec_x[i][j];
				}
				for( int j = 0;j < Lentrg;j ++){
					PF.R_dec_y[i][j] = PF.R_dec_y_y[i][j] / max_j;
					sum += PF.R_dec_y[i][j];
				}
				for(int j = 0;j < Lensrc;j++){
					PF.R_dec_x_s[i][j] = PF.R_dec_x[i][j] / sum;
					//sum += PF.R_dec_x[i][j];
				}
				for( int j = 0;j < Lentrg;j ++){
					PF.R_dec_y_s[i][j] = PF.R_dec_y[i][j] / sum;
					//sum += PF.R_dec_y[i][j];
				}
			}
			for(int i = 0; i < Lentrg; i ++){
				line = bufr.readLine().trim();
				String [] tmp = line.split(" ");
				float max_j  = -1;
				for(int j = 0; j < Lensrc; j ++){
					PF.probs_y[i][j] = Float.parseFloat(tmp[j]); 	
					if(PF.probs_y[i][j] > max_j){
						max_j = PF.probs_y[i][j];
					}
				}
				float sum = 0;
				for(int j = 0; j< Lensrc;j ++){
					PF.probs[i][j] = PF.probs_y[i][j] / max_j;
					sum += PF.probs[i][j];
				}
				for(int j = 0; j< Lensrc;j ++){
					PF.probs_s[i][j] = PF.probs[i][j] / sum;
				}
			}
			
			for(int i = 0; i < Lentrg; i ++){
				line = bufr.readLine().trim();
				String [] tmp = line.split(" ");
				float max_j = -1;
				for(int j = 0; j < Lensrc; j ++){
					PF.trg_x_y[i][j] = Float.parseFloat(tmp[j]);
					if (PF.trg_x_y[i][j] > max_j){
						max_j = PF.trg_x_y[i][j];
					}
				}

				for(int j = 0; j < Lensrc; j ++){
					PF.trg_x[i][j] = PF.trg_x_y[i][j] / max_j;
				}
				
			}
			
			for(int i = 0; i < Lentrg; i ++){
				line = bufr.readLine().trim();
				String [] tmp = line.split(" ");
				float max_j = -1;
				for(int j = 0;j < Lentrg; j ++){
					PF.trg_y_y [i][j] = Float.parseFloat(tmp[j]);
					if (PF.trg_y_y[i][j] > max_j){
						max_j = PF.trg_y_y[i][j];
					}
				}
				
				for(int j = 0; j< Lentrg; j ++){
					PF.trg_y[i][j] = PF.trg_y_y[i][j] / max_j ;					
				}
                
            }
			
			for(int i = 0;i < Lentrg; i ++){
				float max_j = -1;
				for (int j = 0;j < Lensrc; j++){
					if(PF.trg_x_y[i][j] > max_j)
						max_j = PF.trg_x_y[i][j];
				}
				for (int j = 0;j < Lentrg; j++){
					if(PF.trg_y_y[i][j] > max_j)
						max_j = PF.trg_y_y[i][j];
				}
				float sum = 0;
				for(int j = 0;j < Lensrc;j++){
					PF.trg_x[i][j] = PF.trg_x_y[i][j] / max_j;
					sum += PF.trg_x[i][j];
				}
				for( int j = 0;j < Lentrg;j ++){
					PF.trg_y[i][j] = PF.trg_y_y[i][j] / max_j;
					sum += PF.trg_y[i][j];
				}
				for(int j = 0;j < Lensrc;j++){
					PF.trg_x_s[i][j] = PF.trg_x[i][j] / sum;
					//sum += PF.trg_x[i][j];
				}
				for( int j = 0;j < Lentrg;j ++){
					PF.trg_y_s[i][j] = PF.trg_y[i][j] / sum;
					//sum += PF.trg_y[i][j];
				}
			}
			
            PF.jp.repaint();
            PF.jp2.repaint();
            if(index >= 0)
            	bufr.reset();
            //System.out.println("change file");
        }catch(FileNotFoundException e1){
			e1.printStackTrace();
		} catch (IOException e2) {
            e2.printStackTrace();
        }
	}
	
	private float[][] handle(float[][] a){
		int dim1 = a.length;
		int dim2 = a[0].length;
		float maxValue = (float) -2;
		for (int i = 0 ;i < dim1;i ++ ){
			for(int j = 0;j < dim2;j++){
				if(a[i][j] > maxValue && a[i][j] != 1.0){
					maxValue = a[i][j];
				}
			}
		}
	   float [][] res = new float[dim1][dim2];
	   for(int i= 0; i < dim1; i++){
		   for(int j = 0; j < dim2; j ++){
			   if(a[i][j] == 1.0) 
				   res[i][j] = (float)1.0;
			   else{
				   res[i][j] = a[i][j] / maxValue ;
				   
			   }
		   }
	   }
	   
	   return res;
	}
	
	public static void main(String[] args) throws IOException {
		THUMT_Viz paint = new THUMT_Viz();
	}
}


class PaintFrame extends JFrame   {

	public UpPanel jp = new UpPanel();
	public DownPanel jp2 = new DownPanel();
	public JScrollPane jsp1;
	public int sumHeight = 800;
	public JScrollPane jsp2;
	public int total = 0;
	public JButton jb1 = new JButton(">");
    public JButton jb2 = new JButton("<");
    public JTextField jtf = new JTextField(Integer.toString(total + 1),3);
    JPanel p = new JPanel();
    public boolean resizeable = false;
    public JLabel  totalLabel = new JLabel("/   " +Integer.toString(total + 1));
    public JButton jb3 = new JButton(">>");
    public JButton jb4 = new JButton("<<");
	public String srcList[] ;
	public String trgList[];
    public JLabel srcLabels[];
    public JLabel trgLabels[];
	public float [][] R_enc_x_f;
	public float [][] R_enc_x_b;
	public float[][] R_enc_x;
	public float [][] R_ctx_x;
	public float[][] R_dec_x;
	public float[][] R_dec_y;
	public float [][] probs;
	public float [][] trg_x;
	public float [][] trg_y;
	public String fileName;
	public float [][] R_enc_x_f_y;
	public float [][] R_enc_x_b_y;
	public float[][] R_enc_x_y;
	public float [][] R_ctx_x_y;
	public float[][] R_dec_x_y;
	public float[][] R_dec_y_y;
	public float [][] probs_y;
	public float [][] trg_x_y;
	public float [][] trg_y_y;
	
	public float [][] R_enc_x_f_s;
	public float [][] R_enc_x_b_s;
	public float[][] R_enc_x_s;
	public float [][] R_ctx_x_s;
	public float[][] R_dec_x_s;
	public float[][] R_dec_y_s;
	public float [][] probs_s;
	public float [][] trg_x_s;
	public float [][] trg_y_s;
	
	public int left_shift = 320;
	public int top_shift = 0;
	int horizalInterval = 100;
	int verticalInterval = 40;
    int trgLength;
    int srcLength;
    public MyPoint [][]point;

    public void rewrite(){
    	jp.rewrite();
    	jp2.rewrite();
    	
    }
    
    
    
    class MyPoint extends JButton implements MouseListener{

        int radius = 16;
        int layer = 0;
        int index = 0;
        int lock = 0;
        
        MyPoint(){
            super();
            this.setPreferredSize(new Dimension(32, 32));
            this.setContentAreaFilled(false);
            this.setBorderPainted(false);
            this.addMouseListener(this);
        }

        protected void paintComponent(Graphics g){
            super.paintComponent(g);
            int width = this.getPreferredSize().width;
            int height = this.getPreferredSize().height;
            g.fillOval(width / 2 - radius / 2, height / 2 - radius / 2, radius, radius);
            this.setVisible(true);
        }

        public int getX(){
            return this.getLocation().x + radius;
        }
        
        public int getY(){
            return this.getLocation().y + radius;
        }
        
        public int getWidth(){
            return this.getPreferredSize().width;
        }

        @Override
        public void mouseClicked(MouseEvent e) {}

        @Override
        public void mousePressed(MouseEvent e) {
        	jp2.layer = layer;
        	jp2.index = index;
        	jp.repaint();
        	jp2.repaint();          
        }

        @Override
        public void mouseReleased(MouseEvent e) {}

        @Override
        public void mouseEntered(MouseEvent e) {}

        @Override
        public void mouseExited(MouseEvent e) {	}
    }


    class UpPanel extends JPanel implements MouseListener, MouseMotionListener{
        public JLabel labels[] = new JLabel[10];
        public JLabel srcLabels[];
        public JLabel trgLabels[];
        public Rectangle outer_rect = new Rectangle();
        public Rectangle inner_rect = new Rectangle();
        
        int size = 40;
        
        UpPanel(){
            super();
            
            labels[0] = new JLabel("source sentence");
            labels[1] = new JLabel("source word embedding");
            labels[2] = new JLabel("source forward hidden states");
            labels[3] = new JLabel("source backward hidden states");
            labels[4] = new JLabel("source concatenated hidden states");
            labels[5] = new JLabel("attention");
            labels[6] = new JLabel("source context");
            labels[7] = new JLabel("target hidden states");
            labels[8] = new JLabel("target word embedding");
            labels[9] = new JLabel("target sentence");
            labels[0].setBounds(10, 0, 300, 40);
            labels[0].setFont(new java.awt.Font("Dialog", Font.BOLD, 15));
            for (int i = 1; i < 5; ++ i){
                labels[i].setBounds(10, 40 * i + 10, 300, 40);
                labels[i].setFont(new java.awt.Font("Dialog", Font.BOLD, 15));
            }
            labels[5].setBounds(10, 210, 300, 70);
            labels[5].setFont(new java.awt.Font("Dialog", Font.BOLD, 15));
            for (int i=6;i<10;++i){
                labels[i].setBounds(10, 40 * i + 60, 300, 40);
                labels[i].setFont(new java.awt.Font("Dialog", Font.BOLD, 15));
            }
            point =  new MyPoint[8][];
            this.setVisible(true);
            this.addMouseListener(this);
            this.addMouseMotionListener(this);
        }
        
        public void mouseEntered(MouseEvent me) {}

        public void mouseExited(MouseEvent me) {}

        public void mousePressed(MouseEvent me) {
        	
        	System.out.println( me.getY() + " " +  (jsp1.getBounds().height + this.getY()));
        	if(Math.abs(me.getY() -  ( this.getY() + jsp1.getSize().height)) < 30 ){ 
        		resizeable = true;
        		System.out.println("in " +  me.getX() + " " +me.getY());
        	}	
        	else{
        		resizeable = false;
        	}
        }

        public void mouseReleased(MouseEvent me) {
        	this.setCursor(new Cursor(Cursor.DEFAULT_CURSOR));
        }

        public void mouseClicked(MouseEvent me) {}

        public void mouseMoved(MouseEvent me) {
        	if ((jp.outer_rect.contains(me.getPoint())) && (!jp.inner_rect.contains(me.getPoint()))){
        		
        	}
        		//this.setCursor(new Cursor(Cursor.MOVE_CURSOR));
        	else
        		this.setCursor(new Cursor(Cursor.DEFAULT_CURSOR));
        }

        public void mouseDragged(MouseEvent me) {
        	if (resizeable){
        		
        		System.out.println("dragged : " +  (me.getX()) + " " + (me.getY()));
        		System.out.println("jsp2 " + jsp1.getY());
        		int height1 = me.getY() - jsp1.getY();
        		int height2 = jsp2.getY() + jsp2.getHeight() - me.getY();
        		double ratio1 = (double)height1 / (height1 +height2);
        		double ratio2 = (double)height2 / (height1 +height2);
        		System.out.println(ratio1);
        		
        		GridBagLayout layout = new GridBagLayout();
        		PaintFrame.this.setLayout(layout);
        		GridBagConstraints s= new GridBagConstraints();
        		s.fill = GridBagConstraints.BOTH;
                s.gridwidth = 1;
                s.gridheight = 1;
                s.weightx = 1;
                s.weighty = 0;
                layout.setConstraints(p, s);
                s.gridy = 1;
                s.gridheight = 7;
                s.weightx = 1;
                s.weighty = ratio1;
                layout.setConstraints(jsp1, s);
                s.gridy = 8;
                s.gridheight = 3;
                s.weightx = 1;
                s.weighty = ratio2;
                layout.setConstraints(jsp2, s);
               
                Dimension screensize = Toolkit.getDefaultToolkit().getScreenSize();
              
                PaintFrame.this.setSize(screensize.width + 5, screensize.height);
                PaintFrame.this.setExtendedState(Frame.MAXIMIZED_BOTH);
                PaintFrame.this.setVisible(true);
               
                
        	}
        }
        
        public void rewrite(){
        	repaint();
        }
        
        public void paintComponent(Graphics g){
            super.paintComponent(g);
            inner_rect = this.getVisibleRect();
            outer_rect.setRect(inner_rect.x+1, inner_rect.y+1, inner_rect.width, inner_rect.height + 100);
            outer_rect.setRect(inner_rect.x+1, inner_rect.y+1, inner_rect.width, inner_rect.height + 150);
            
            this.removeAll();
            this.setLayout(null);
            if(srcList == null || trgList == null)
                return;
            int pointNumber = 0;
            if (srcList.length > trgList.length)
            	pointNumber = srcList.length;
            else 
            	pointNumber = trgList.length;
            this.setPreferredSize(new Dimension(left_shift + horizalInterval * pointNumber + 23, 600));
            
            srcLength = srcList.length;
            for(int i = 0; i < 4; ++ i) {
                point[i] = new MyPoint[srcLength];
                for(int j = 0; j < srcLength; ++ j){
                    point[i][j] = new MyPoint();
                    point[i][j].layer = i;
                    point[i][j].index = j;
                }
            }
            trgLength = trgList.length;
            for(int i = 4; i < 8; ++ i) {
                point[i] = new MyPoint[trgLength];
                for(int j = 0;j < trgLength; ++ j){
                    point[i][j] = new MyPoint();
                    point[i][j].layer = i;
                    point[i][j].index = j;
                }
            }
            srcLabels = new JLabel[srcLength];
            trgLabels = new JLabel[trgLength];

            for (int i=0;i<srcLength;++i) {
                srcLabels[i] = new JLabel(srcList[i], JLabel.CENTER);
                //srcLabels[i].setPreferredSize(new Dimension(70, 40));
                srcLabels[i].setBounds(left_shift + horizalInterval * i - 3, top_shift, 70, 40);
                srcLabels[i].setHorizontalAlignment(SwingConstants.CENTER);
                srcLabels[i].setFont(new java.awt.Font("Dialog", Font.BOLD, 15));
                srcLabels[i].repaint();
                this.add(srcLabels[i]);
				//word embedding
                point[0][i].setBounds(left_shift + horizalInterval * i,top_shift + verticalInterval, size, size);
				//forward hidden states
                point[1][i].setBounds(left_shift + horizalInterval * i, top_shift + verticalInterval * 2 + 5, size, size);
                point[2][i].setBounds(left_shift + horizalInterval * i, top_shift + verticalInterval * 3 - 5, size, size);
				//concatenated hidden states
                point[3][i].setBounds(left_shift + horizalInterval * i, top_shift + verticalInterval * 4, size, size);
                point[0][i].setBorderPainted(false);
                point[1][i].setBorderPainted(false);
                point[2][i].setBorderPainted(false);
                point[3][i].setBorderPainted(false);
                this.add(point[0][i]);
                this.add(point[1][i]);
                this.add(point[2][i]);
                this.add(point[3][i]);
            }
          
            for(int i = 0; i < trgLength; ++ i){
                point[4][i].setBounds(left_shift + horizalInterval * i,top_shift + verticalInterval * 5 + 13, size, size);
                point[4][i].setBorderPainted(false);
                this.add(point[4][i]);
            }
            //source context
            for(int i = 0;i < trgLength;++i){
                point[5][i].setBounds(left_shift + horizalInterval * i, top_shift + verticalInterval * 7 + 5, size, size);
                point[5][i].setBorderPainted(false);
                this.add(point[5][i]);
            }
            for(int i=0;i<trgLength;++i){
                //target hidden states
                point[6][i].setBounds(left_shift + horizalInterval * i, top_shift + verticalInterval * 8 + 5, size, size);
                point[7][i].setBounds(left_shift + horizalInterval * i, top_shift + verticalInterval * 9 + 5, size, size);
                point[6][i].setBorderPainted(false);
                point[7][i].setBorderPainted(false);
                trgLabels[i] = new JLabel(trgList[i], JLabel.CENTER);
                trgLabels[i].setBounds(left_shift + horizalInterval * i - 18, top_shift + verticalInterval * 9 + 60, 100, 40);
                trgLabels[i].setHorizontalAlignment(SwingConstants.CENTER);
                trgLabels[i].setFont(new java.awt.Font("Dialog", Font.BOLD, 14));
                this.add(trgLabels[i]);
                this.add(point[6][i]);
                this.add(point[7][i]);
            }
			for(int i=0;i<10;++i) {
                this.add(labels[i]);
            }

            ((Graphics2D)g).setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            for(int i = 0; i < srcLength - 1; ++ i){
                drawAL(point[1][i].getX() + 26, point[1][i].getY() + 16, point[1][i + 1].getX() + 5,point[1][i + 1].getY() + 16, (Graphics2D)g);
                drawAL(point[2][i + 1].getX() + 5, point[2][i].getY() + 16, point[2][i].getX() + 26,point[2][i + 1].getY() + 16, (Graphics2D)g);
                drawCurve(point[0][i].getX(), point[0][i].getY() + 16,point[2][i].getX() + 23,point[2][i].getY() + 13, 40, 71, -95, 180, (Graphics2D)g);
                drawCurve(point[1][i].getX() - 10,point[1][i].getY() + 20,point[3][i].getX() + 7,point[3][i].getY() + 17, 40, 71, 95, 180, (Graphics2D)g);
            }

            for(int i = 0;i < trgLength - 1; ++ i){
                drawAL(point[6][i].getX() + 26,point[6][i].getY() + 16,point[6][i + 1].getX()+5,point[6][i+1].getY()+16,(Graphics2D)g);
                drawAL(point[7][i].getX() + 25,point[7][i].getY() + 14,point[6][i + 1].getX()+8,point[6][i+1].getY()+22,(Graphics2D)g);
            }

            for(int i=0;i<srcLength;++ i){
                drawAL(point[0][i].getX() + 16, point[0][i].getY() + 25, point[1][i].getX() + 16, point[1][i].getY() + 8, (Graphics2D)g);
                drawAL(point[1][i].getX() + 16, point[1][i].getY() + 25, point[2][i].getX() + 16, point[2][i].getY() + 8, (Graphics2D)g);
                drawAL(point[2][i].getX() + 16, point[2][i].getY() + 25, point[3][i].getX() + 16, point[3][i].getY() + 8, (Graphics2D)g);
            }

            for(int i=0;i<trgLength;++i){
                drawAL(point[5][i].getX() + 16, point[5][i].getY() + 25, point[6][i].getX() + 16,point[6][i].getY() + 8, (Graphics2D)g);
                drawAL(point[6][i].getX() + 16, point[6][i].getY() + 25, point[7][i].getX() + 16,point[7][i].getY() + 8, (Graphics2D)g);
            }


            Stroke old_stroke = ((Graphics2D)g).getStroke();
            Stroke stroke = new BasicStroke(2.5f, BasicStroke.CAP_BUTT ,BasicStroke.JOIN_ROUND,
                    3.5f, new float[]{15,10,}, 0f);
            ((Graphics2D)g).setStroke(stroke);
            ((Graphics2D)g).drawRect(left_shift, 230, trgList.length * horizalInterval - 20	, 30);
            ((Graphics2D)g).drawRect(left_shift, 303, trgList.length * horizalInterval - 20	, 30);
            ((Graphics2D)g).setStroke(old_stroke);
            this.setVisible(true);
            int layer = jp2.layer;
            int index = jp2.index;
            if(layer >= 0 && index >= 0){
            	point[layer][index].setBorderPainted(true);
            }
        }
    }

    class DownPanel extends JPanel{
        int layer = 0;
        int index = 0;
        int dis = 80;
        public JButton jb1 = new JButton("next");
        
        DownPanel(){
            super();
            this.setVisible(true);
           
        }
        
        public void rewrite(){
        	repaint();
        }
        
        private void drawRect(Color[]colors, Graphics2D gh2, int y){
            int i;
            int offset = 200;
            gh2.setColor(colors[0]);
            gh2.fillRect(10 + offset, y, dis/2, 10);
            for(i = 0; i < colors.length - 1; ++ i){
                Paint  paint=new GradientPaint((i) * dis + 50 + offset, y, colors[i], (i + 1) * dis + 50 + offset, y, colors[i + 1], false);
                gh2.setPaint(paint);
                gh2.fillRect(i * dis + 50 + offset, y, dis, 10);
            }
            gh2.setColor(colors[i]);
            gh2.fillRect(i * dis + 50 + offset, y, dis / 2, 10);
        }

        public void paint(Graphics g) {
        	int offset_for_word  = 200;
        	DecimalFormat decimalFormat=new DecimalFormat("0.0000");
        	DecimalFormat decimalFormat2=new DecimalFormat("0.0000");
            super.paint(g);
            if (srcList == null || trgList == null)
                return;
            Graphics2D gh2 = (Graphics2D) g;
            int i = 0;
            Color[] colors;
            switch (layer) {
                case 0: {
                	
                	gh2.drawString("source contextual words", 3, 10);
                	//gh2.drawString("relevance", 3, 30);
                    colors = new Color[srcList.length];
                    for (i = 0; i < srcList.length; i ++) {
                        if (i != index) {
                            gh2.setColor(Color.white);
                            colors[i] = Color.white;
                        } else {
                            gh2.setColor(Color.blue);
                            colors[i] = Color.blue;
                        }
                        gh2.setColor(Color.black);
                        int offset = 30 + offset_for_word;
                        if(srcList[i].length() == 1)
                        	offset = 38 + offset_for_word;
                        gh2.drawString(srcList[i], i * dis + offset, 10);
                    }
                    drawRect(colors, gh2,30);
                    break;
                }
                case 1:{
                	gh2.drawString("source contextual words", 3, 10);
                	//gh2.drawString("relevance", 3, 30);
                	gh2.drawString("original relevance", 3, 60);
                	gh2.drawString("normalized relevance", 3, 90);
                    colors = new Color[srcList.length];
                    for (i = 0; i < srcList.length; i ++) {
                        gh2.setColor(Color.black);
                        int offset = 30 + offset_for_word;
                        if(srcList[i].length() == 1)
                        	offset = 38 + offset_for_word; 
                        gh2.drawString(srcList[i], i * dis + offset, 10);
                        double s = (double) ((int) (R_enc_x_f[index][i] * 100000)) / 100000;
                        
                        if (i > index) {
                            colors[i] = Color.white;
                            s = 0;
                        } else {
                            int rgb = (int) ((1 - R_enc_x_f[index][i]) * 255) % 256;
                            colors[i] = (new Color(rgb, rgb, 255));
                        }
                        gh2.setColor(Color.black);
                        s = (double) ((int) (R_enc_x_f_y[index][i] * 100000)) / 100000;
                        String ss = decimalFormat2.format(s);
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 60);
                        s = (double) ((int) (R_enc_x_f_s[index][i] * 100000)) / 100000;
                        ss = decimalFormat.format(s);
                        
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 90);
                    }
                    drawRect(colors, gh2,30);
                    break;
                }
                case 2: {
                	gh2.drawString("source contextual words", 3, 10);
                	
                	gh2.drawString("original relevance", 3, 60);
                	gh2.drawString("normalized relevance", 3, 90);
                    colors = new Color[srcList.length];
                    for (i = 0; i < srcList.length; i++) {
                        gh2.setColor(Color.black);
                        int offset = 30 + offset_for_word;
                        if(srcList[i].length() == 1)
                        	offset = 38 + offset_for_word;
                        gh2.drawString(srcList[i], i * dis + offset, 10);
                        double s = (double) ((int) (R_enc_x_b[index][i] * 100000)) / 100000;
                        if (i < index) {
                            colors[i] = Color.white;
                            s = 0.0;
                        } else {
                            int rgb = (int) ((1 - R_enc_x_b[index][i]) * 255) % 256;
                            colors[i] = (new Color(rgb, rgb, 255));
                        }
                        gh2.setColor(Color.black);
                        s = (double) ((int) (R_enc_x_b_y[index][i] * 100000)) / 100000;
                        String ss = decimalFormat2.format(s);
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 60);
                        s = (double) ((int) (R_enc_x_b_s[index][i] * 100000)) / 100000;
                        ss = decimalFormat.format(s);
                        
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 90);
                    }
                    drawRect(colors, gh2,30);
                    break;
                }
                case 3:{
                	gh2.drawString("source contextual words", 3, 10);
                	//gh2.drawString("relevance", 3, 30);
                	gh2.drawString("original relevance", 3, 60);
                	gh2.drawString("normalized relevance", 3, 90);
                    colors = new Color[srcList.length];
                    for (i = 0; i < srcList.length; i++) {
                        gh2.setColor(Color.black);
                        int offset = 30 + offset_for_word;
                        if(srcList[i].length() == 1)
                        	offset = 38 + offset_for_word;
                        gh2.drawString(srcList[i], i * dis + offset, 10);
                        double s = (double) ((int) (R_enc_x[index][i] * 100000)) / 100000;
                        int rgb = (int) ((1 - R_enc_x[index][i]) * 255) % 256;
                        colors[i] = (new Color(rgb, rgb, 255));
                        gh2.setColor(Color.black);
                        s = (double) ((int) (R_enc_x_y[index][i] * 100000)) / 100000;
                        String ss = decimalFormat2.format(s);
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 60);
                        s = (double) ((int) (R_enc_x_s[index][i] * 100000)) / 100000;
                        ss = decimalFormat.format(s);
                        
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 90);
                    }
                    drawRect(colors, gh2,30);
                    break;
                }
                case 6:{
                	gh2.drawString("source contextual words", 3, 10);
                	//gh2.drawString("relevance", 3, 30);
                	gh2.drawString("original relevance", 3, 60);
                	gh2.drawString("normalized relevance", 3, 90);
                	if(index != 0){
	                	gh2.drawString("target contextual words", 3, 120);
	                //	gh2.drawString("relevance", 3, 130);
	                	gh2.drawString("original relevance", 3, 160);
	                	gh2.drawString("normalized relevance", 3, 190);
                	}
                    colors = new Color[srcList.length];
                    for (i = 0; i < srcList.length; i++) {
                        gh2.setColor(Color.black);
                        int rgb;
                        int offset = 30 + offset_for_word;
                        if(srcList[i].length() == 1)
                        	offset = 38 + offset_for_word;
                        gh2.drawString(srcList[i], i * dis + offset, 10);
                        rgb = (int) ((1 - R_dec_x[index][i]) * 255) % 256;
                        colors[i] = new Color(rgb, rgb, 255);
                        gh2.setColor(Color.black);
                        double s = (double) ((int) (R_dec_x_y[index][i] * 100000)) / 100000;
                        String ss = decimalFormat2.format(s);
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 60);
                        s = (double) ((int) (R_dec_x_s[index][i] * 100000)) / 100000;
                        ss = decimalFormat.format(s);
                        
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 90);
                    }
                    drawRect(colors,gh2,30);
                    
                    if(index != 0){
                        colors = new Color[index];
                       // System.out.println(index);
                        for (i = 0; i < index; i++) {
                            gh2.setColor(Color.black);
                            int offset = 30 + offset_for_word;
                            gh2.drawString(trgList[i], (i) * dis + offset, 120);
                            int rgb = (int) ((1 - R_dec_y[index][i]) * 255) % 256;
                            colors[i] = new Color(rgb, rgb, 255);
                            double s = (double) ((int) (R_dec_y[index][i] * 100000)) / 100000;
                            gh2.setColor(Color.black);
                            s = (double) ((int) (R_dec_y_y[index][i] * 100000)) / 100000;
                            String ss = decimalFormat2.format(s);
                            gh2.drawString(ss, i * dis + 25  + offset_for_word, 160);
                            s = (double) ((int) (R_dec_y_s[index][i] * 100000)) / 100000;
                            ss = decimalFormat.format(s);
                            
                            gh2.drawString(ss, i * dis + 25 + offset_for_word, 190);
                        }
                        drawRect(colors,gh2,130);
                    }
                    break;
                }
                case 7:{
                	gh2.drawString("source contextual words", 3, 10);
                	//gh2.drawString("relevance", 3, 30);
                	gh2.drawString("original relevance", 3, 60);
                	gh2.drawString("normalized relevance", 3, 90);
                	if(index != 0){
	                	gh2.drawString("target contextual words", 3, 120);
	                //	gh2.drawString("relevance", 3, 130);
	                	gh2.drawString("original relevance", 3, 160);
	                	gh2.drawString("normalized relevance", 3, 190);
                	}
                    colors = new Color[srcList.length];
                    for (i = 0; i < srcList.length; i++) {
                        gh2.setColor(Color.black);
                        int rgb;
                        int offset = 30 + offset_for_word;
                        if(srcList[i].length() == 1)
                        	offset = 38 + offset_for_word;
                        gh2.drawString(srcList[i], i * dis + offset, 10);
                        double s;
                        if (index + 1 == trgList.length) {
                            s = (double) ((int) ( trg_x[index][i]  * 100000)) / 100000;
                        } else
                            s = (double) ((int) (trg_x[index][i] * 100000)) / 100000;
                        if (index + 1 == trgList.length) {
                            rgb = (int) ((1 - s) * 255) % 256;
                        } else
                            rgb = (int) ((1 - trg_x[index][i]) * 255) % 256;

                        colors[i] = new Color(rgb, rgb, 255);
                        gh2.setColor(Color.black);
                        s = (double) ((int) (trg_x_y[index][i] * 100000)) / 100000;
                        String ss = decimalFormat2.format(s);
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 60);
                        s = (double) ((int) (trg_x_s[index][i] * 100000)) / 100000;
                        ss = decimalFormat.format(s);
                        
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 90);
                    }
                    drawRect(colors,gh2,30);
                    if(index !=0){
                        colors = new Color[index];
                        for (i = 0; i < index; i++) {
                            gh2.setColor(Color.black);
                            int offset = 30 + offset_for_word; 
                            gh2.drawString(trgList[i], (i) * dis + offset, 120);
                            int rgb = (int) ((1 - trg_y[index][i]) * 255) % 256;
                            colors[i] = new Color(rgb, rgb, 255);
                            double s = (double) ((int) (trg_y[index][i] * 100000)) / 100000;
                            gh2.setColor(Color.black);
                            s = (double) ((int) (trg_y_y[index][i] * 100000)) / 100000;
                            String ss = decimalFormat2.format(s) ;
                            gh2.drawString(ss, i * dis + 25 + offset_for_word, 160);
                            s = (double) ((int) (trg_y_s[index][i] * 100000)) / 100000;
                            ss = decimalFormat.format(s); 
                            gh2.drawString(ss, i * dis + 25 + offset_for_word, 190);
                        }
                        drawRect(colors,gh2,130);
                    }
                    break;
                }
                case 5:{
                	gh2.drawString("source contextual words", 3, 10);
                	//gh2.drawString("relevance", 3, 30);
                	gh2.drawString("original relevance", 3, 60);
                	gh2.drawString("normalized relevance", 3, 90);
                    colors = new Color[srcList.length];
                    for (i = 0; i < srcList.length; i++) {
                        gh2.setColor(Color.black);
                        int offset = 30 + offset_for_word;
                        if(srcList[i].length() == 1)
                        	offset = 38 + offset_for_word;
                        gh2.drawString(srcList[i], i * dis + offset, 10);
                        double s;
                        int rgb;
                        s = (double) ((int) (R_ctx_x[index][i] * 100000)) / 100000;
                        rgb = (int) ((1 - R_ctx_x[index][i]) * 255) % 256;
                        colors[i] = new Color(rgb, rgb, 255);
                        gh2.setColor(Color.black);
                        s = (double) ((int) (R_ctx_x_y[index][i] * 100000)) / 100000;
                        String ss = decimalFormat2.format(s);
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 60);
                        s = (double) ((int) (R_ctx_x_s[index][i] * 100000)) / 100000;
                        ss = decimalFormat.format(s);
                        
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 90);
                    }
                    drawRect(colors,gh2,30);
                    break;
                }
                case 4:{
                	gh2.drawString("source contextual words", 3, 10);
                	//gh2.drawString("attention", 3, 30);
                	gh2.drawString("original attention", 3, 60);
                	gh2.drawString("normalized attention", 3, 90);
                    colors = new Color[srcList.length];
                    for (i = 0; i < srcList.length; i++) {
                        gh2.setColor(Color.black);
                        int offset = 30 + offset_for_word;
                        if(srcList[i].length() == 1)
                        	offset = 38 + offset_for_word;
                        gh2.drawString(srcList[i], i * dis + offset, 10);
                        double s = (double) ((int) (probs[index][i] * 100000)) / 100000;
                        int rgb = (int) ((1 - probs[index][i]) * 255) % 256;

                        colors[i] = new Color(rgb, rgb, 255);
                        gh2.setColor(Color.black);
                        s = (double) ((int) (probs_s[index][i] * 100000)) / 100000;
                        String ss = decimalFormat2.format(s);
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 60);
                        s = (double) ((int) (probs_y[index][i] * 100000)) / 100000;
                        ss = decimalFormat.format(s);
                        
                        gh2.drawString(ss, i * dis + 25 + offset_for_word, 90);
                    }
                    drawRect(colors,gh2,30);
                    break;
                }
            }
        }
    }

    PaintFrame(String title) {
        super(title);
        jp.setPreferredSize(new Dimension(4000, 600));
        jp.setMinimumSize(new Dimension(100, 100));
        jsp1  = new JScrollPane(jp,
                ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS,
                ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        GridBagLayout layout = new GridBagLayout();
        jsp1.setPreferredSize(new Dimension(10000,600));
        this.setLayout(layout);
        
        p.setPreferredSize(new Dimension(4000,50));
        p.setLayout(new FlowLayout(1, 8, 8));
        p.add(jb4);
        p.add(jb2);
        p.add(jtf);
        jtf.setHorizontalAlignment(JTextField.RIGHT);
        p.add(totalLabel);
        p.add(jb1);
        p.add(jb3);

        this.add( p);
        this.add( jsp1);
        
        jp2.setPreferredSize(new Dimension(4000, 200));
        jp2.setMinimumSize(new Dimension(100, 100));
        jsp2  = new JScrollPane(jp2,
                ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS,
                ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        this.add( jsp2);
        
        GridBagConstraints s= new GridBagConstraints();
        s.fill = GridBagConstraints.BOTH;
        s.gridwidth = 1;
        s.gridheight = 1;
        s.weightx = 1;
        s.weighty = 0;
        layout.setConstraints(p, s);
        s.gridy = 1;
        s.gridheight = 7;
        s.weightx = 1;
        s.weighty = 2.3;
        layout.setConstraints(jsp1, s);
        s.gridy = 8;
        s.gridheight = 3;
        s.weightx = 1;
        s.weighty = 1;
        layout.setConstraints(jsp2, s);
        Dimension screensize = Toolkit.getDefaultToolkit().getScreenSize();
        System.out.println("sumHeight : " + sumHeight);
        this.setSize(screensize.width, screensize.height);
        this.setExtendedState(Frame.MAXIMIZED_BOTH);
        this.setVisible(true);
        
        this.setDefaultCloseOperation(this.EXIT_ON_CLOSE);
        sumHeight = jp.getSize().height + jp2.getSize().height;
    }
	public static void drawCurve(int sx, int sy, int ex, int ey,int width,int height,int startAngle,int arcAngle,Graphics2D g2){
		
        double H =  3;  
        double L =  3; 
        int x3 =  0;
        int y3 =  0;
        int x4 =  0;
        int y4 =  0;
        double awrad = Math.atan(L / H);  
        int sxx = ex + width;
        int syy = sy + (int)(height / 5);
        if(startAngle > 0){
        	sxx = ex - width; 
        }
        double arraow_len = Math.sqrt(L * L + H * H);    
        double [] arrXY_1 = rotateVec(ex - sxx, ey - syy, awrad, true, arraow_len);
        double [] arrXY_2 = rotateVec(ex - sxx, ey - syy,  -awrad, true, arraow_len);
        double x_3 = ex - arrXY_1[0];   
        double y_3 = ey - arrXY_1[1];
        double x_4 = ex - arrXY_2[0];   
        double y_4 = ey - arrXY_2[1];

        Double X3 = new  Double(x_3);
        x3  =  X3.intValue();
        Double Y3 = new  Double(y_3);
        y3  =  Y3.intValue();
        Double X4 = new  Double(x_4);
        x4  =  X4.intValue();
        Double Y4 = new  Double(y_4);
        y4 = Y4.intValue();

        g2.drawArc(sx, sy, width, height, startAngle, arcAngle);
        g2.drawLine(ex, ey, x3, y3);
        g2.drawLine(ex, ey, x4, y4);
	}
	
	public static void drawAL(int sx, int sy, int ex, int ey,Graphics2D g2)
	{
	       double H =  3 ;  
	       double L =  3 ;
	       int x3 =  0 ;
	       int y3 =  0 ;
	       int x4 =  0 ;
	       int y4 =  0 ;
	       double awrad = Math.atan(L / H);     
	       double arraow_len = Math.sqrt(L * L + H * H);   
	       double [] arrXY_1 = rotateVec(ex - sx, ey - sy, awrad, true, arraow_len);
	       double [] arrXY_2 = rotateVec(ex - sx, ey - sy,  -awrad, true, arraow_len);
	       double x_3 = ex - arrXY_1[0];  
	       double y_3 = ey - arrXY_1[1];
	       double x_4 = ex - arrXY_2[0]; 
	       double y_4 = ey - arrXY_2[1];

	       Double X3 = new Double(x_3);
	       x3 = X3.intValue();
	       Double Y3 = new Double(y_3);
	       y3 = Y3.intValue();
	       Double X4 = new Double(x_4);
	       x4 = X4.intValue();
	       Double Y4 = new Double(y_4);
	       y4 = Y4.intValue();
	       g2.drawLine(sx, sy, ex, ey);
	       g2.drawLine(ex, ey, x3, y3);
	       g2.drawLine(ex, ey, x4, y4);
	     }	    
	
	public  static double [] rotateVec(int px, int py, double ang, boolean isChLen, double newLen){
        double mathstr[] = new double[2];
        double vx = px * Math.cos(ang) - py * Math.sin(ang);
        double vy = px * Math.sin(ang) + py * Math.cos(ang);
        if(isChLen){
           double d  =  Math.sqrt(vx * vx + vy * vy);
           vx = vx / d * newLen;
           vy = vy / d * newLen;
           mathstr[0] = vx;
           mathstr[1] = vy;
       } 
       return  mathstr;
	}

}